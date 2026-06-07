import os
import sys
import time
import json
import argparse
import datetime
import pandas as pd
import re
import yaml
import torch

# 导入核心模块
from core.state_machine import JobAgentStateMachine
from core.retriever import AdvancedRetriever

# ==========================================
# 1. 路径与配置设定
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "_checkpoints")


def load_config():
    """安全加载配置文件"""
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    return {}


def load_occupation_corpus():
    """从官方 CSV 中提取并构建全量特征的语料库 (包含主要工作任务)"""
    print(f"⏳ 正在读取大典构建基础语料库...")
    dict_csv_path = os.path.join(PROJECT_ROOT, "data", "raw_jd", "2022年职业分类大典（整体修订）.csv")
    if not os.path.exists(dict_csv_path):
        raise FileNotFoundError(f"找不到职业大典源文件: {dict_csv_path}")

    df = pd.read_csv(dict_csv_path).fillna("")
    corpus = []

    for _, row in df.iterrows():
        code = str(row['职业编码']).strip()
        dash_count = code.count('-')

        # 只提取三级(大类)和四级(细分)
        if dash_count in [2, 3]:
            name = str(row.get('职业名称', '')).strip()
            desc = re.sub(r'\s+', ' ', str(row.get('职业描述', '')).strip())
            tasks = re.sub(r'\s+', ' ', str(row.get('主要工作任务', '')).strip())

            prefix = "【细分四级】" if dash_count == 3 else "【三级大类】"

            # 拼装全量文本：名称 + 描述 + 主要工作任务
            text = f"{prefix}{name}：{desc}"
            if tasks:
                text += f" 主要工作任务：{tasks}"

            corpus.append({
                "code": code,
                "name": name,
                "text": text
            })

    print(f"✅ 向量语料库构建完毕，共提取 {len(corpus)} 条记录 (已挂载全量工作任务)。")
    return corpus


def resolve_test_files(config):
    """
    解析测试文件路径（支持多文件）
    优先级：
      1. 命令行 --files 参数
      2. config.yaml 中的 test_csv_path（支持字符串或列表）
      3. 默认路径 data/raw_jd/ 下所有 CSV
    """
    # 优先使用命令行参数（在 main() 中通过 parser 传入）
    # 此处只处理 config 中的配置
    raw = config.get("test", {}).get("test_csv_path", [])

    if isinstance(raw, str):
        paths = [raw]
    elif isinstance(raw, list):
        paths = raw
    else:
        # 默认：扫描 data/raw_jd/ 下所有 CSV
        default_dir = os.path.join(PROJECT_ROOT, "data", "raw_jd")
        if os.path.isdir(default_dir):
            paths = [os.path.join(default_dir, f) for f in os.listdir(default_dir) if f.endswith('.csv')]
        else:
            paths = []

    # 展开为绝对路径，过滤不存在的文件
    resolved = []
    for p in paths:
        absp = p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)
        if os.path.exists(absp):
            resolved.append(absp)
        else:
            print(f"⚠️ 警告：测试文件不存在，已跳过: {absp}")

    return resolved


def save_checkpoint(path, state):
    """保存断点"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"  💾 断点已保存: {path}")


def load_checkpoint(path):
    """加载断点，返回 state dict 或 None"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        print(f"  🔄 检测到断点文件，将从第 {state.get('last_index', 0) + 1} 条继续...")
        return state
    except Exception as e:
        print(f"  ⚠️ 断点文件读取失败: {e}，将从头开始")
        return None


def process_row(index, row, agent, retriever, results_list, status_counts, task_hit_l3, task_hit_l2):
    """处理单行 JD 数据，返回 (results_list, status_counts, task_hit_l3, task_hit_l2)"""
    row_id = row.get('_id', f"IDX_{index}")
    raw_name = str(row.get('job_name', '')).strip()
    raw_desc = str(row.get('job_descrip', '')).strip()

    # 提取真实标签 (Ground Truth)
    acceptable_truths = set()
    for col_name in ['occupation_code', 'occupation_code1', 'occupation_code2', 'occupation_code3', '职业备选项']:
        code_val = str(row.get(col_name, '')).strip()
        if pd.notna(code_val) and code_val.lower() not in ['-9', '-8', 'nan', 'none', '']:
            acceptable_truths.update(re.findall(r'\d-\d{2}-\d{2}', code_val))

    if not acceptable_truths:
        print(f"⚠️ 跳过第 {index+1} 条：无有效真实标签")
        return results_list, status_counts, task_hit_l3, task_hit_l2

    truth_occ_code = ", ".join(list(acceptable_truths))
    acceptable_truths_lvl2 = {"-".join(c.split('-')[:2]) for c in acceptable_truths if "-" in c}

    print(f"\n▶️ [{index+1}] 正在处理: 【{raw_name}】")
    start_t = time.time()

    try:
        print("  🔍 正在通过双轨引擎进行全域扫描与图谱安检...")
        initial_context = retriever.retrieve(raw_name, raw_desc)

        torch.cuda.empty_cache()

        result_dict = agent.run(raw_name, raw_desc, initial_context)
        final_state = result_dict.get("status", "ERROR")
        predicted_code = result_dict.get("code", "未提取")
        confidence_score = result_dict.get("confidence", 0)
        reasoning_log = json.dumps(result_dict.get("reasoning", {}), ensure_ascii=False)
        trajectory_log = "\n\n===HOP===\n\n".join(result_dict.get("log", []))

    except Exception as e:
        print(f"  ❌ 处理异常: {e}")
        final_state = "ERROR"
        predicted_code = "未提取"
        confidence_score = 0
        reasoning_log = f"运行报错: {str(e)}"
        trajectory_log = ""

    # 计算命中率 (兼容大模型越级预测到四级细分的情况)
    if predicted_code not in ["未提取", "ERROR", "MANUAL_REVIEW", "8-00-00"]:
        parts = predicted_code.split('-')
        predicted_lvl3 = "-".join(parts[:3]) if len(parts) >= 3 else predicted_code
        predicted_lvl2 = "-".join(parts[:2]) if len(parts) >= 2 else predicted_code

        is_hit_l3 = predicted_lvl3 in acceptable_truths
        is_hit_l2 = predicted_lvl2 in acceptable_truths_lvl2
    else:
        is_hit_l3 = False
        is_hit_l2 = False

    if is_hit_l3: task_hit_l3 += 1
    if is_hit_l2: task_hit_l2 += 1

    # P0-F: 按状态分类独立计数
    if final_state in status_counts:
        status_counts[final_state] += 1
    else:
        status_counts["ERROR"] += 1

    print(f"  [耗时: {time.time() - start_t:.1f}s]")
    print(f"  🏆 真实标签池 (三级): {truth_occ_code}")
    print(f"  🤖 预测结果: {predicted_code} (状态: {final_state}, 置信度: {confidence_score})")
    print(f"  🏅 三级命中: {'✅' if is_hit_l3 else '❌'}  |  二级命中: {'✅' if is_hit_l2 else '❌'}")

    results_list.append({
        "数据ID": row_id,
        "原始名称": raw_name,
        "原始职责": raw_desc[:100] + "...",
        "真实标签(三级)": truth_occ_code,
        "模型预测(三级)": predicted_code,
        "Top-1命中(三级)": is_hit_l3,
        "Top-1命中(二级)": is_hit_l2,
        "状态机终局": final_state,
        "置信度评分": confidence_score,
        "Agent推理日志": reasoning_log,
        "完整推理轨迹": trajectory_log
    })

    return results_list, status_counts, task_hit_l3, task_hit_l2


def main():
    parser = argparse.ArgumentParser(description="Job Agent 批量测试框架")
    parser.add_argument("--files", nargs='+', help="指定测试 CSV 文件列表（覆盖 config 配置）")
    parser.add_argument("--resume", type=str, default=None, help="从指定断点文件恢复（传入 checkpoint JSON 路径）")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="每 N 条保存一次断点（默认 10）")
    parser.add_argument("--max-records", type=int, default=None, help="最多处理 N 条（用于快速测试）")
    args = parser.parse_args()

    print("="*60)
    print("🚀 Job Agent (Dual-Track RAG Edition) - 批量测试框架启动中...")
    print("="*60)

    # 1. 加载配置与动态语料
    config = load_config()
    occupation_corpus = load_occupation_corpus()

    # 2. 解析测试文件列表
    if args.files:
        test_files = [os.path.abspath(f) for f in args.files if os.path.exists(f)]
        if not test_files:
            print(f"❌ 命令行指定的文件均不存在: {args.files}")
            return
    else:
        test_files = resolve_test_files(config)
        if not test_files:
            print(f"❌ 未找到任何测试文件，请检查 config.yaml 或 --files 参数")
            return

    print(f"📂 待测试文件 ({len(test_files)} 个):")
    for f in test_files:
        print(f"    - {f}")

    # 3. 初始化核心组件（只初始化一次）
    print("\n[模块加载 1/2] 正在启动双轨融合引擎 (BGE-M3 + GraphRAG) ...")
    retriever = AdvancedRetriever(config, occupation_corpus)

    print("\n[模块加载 2/2] 正在启动 Agent 状态机...")

    from core.llm_client import Gemma4Client
    from prompts.meta_rules import SYSTEM_RULES_PROMPT
    from prompts.templates import JSON_COT_TEMPLATE

    llm_client = Gemma4Client(config)

    agent = JobAgentStateMachine(
        llm_client=llm_client,
        retriever=retriever,
        config=config,
        rules_prompt=SYSTEM_RULES_PROMPT,
        json_template=JSON_COT_TEMPLATE
    )

    # 4. 断点续传状态
    checkpoint_path = None
    results_list = []
    task_hit_l3 = 0
    task_hit_l2 = 0
    status_counts = {
        "SUCCESS": 0,
        "REVIEW_SUGGESTED": 0,
        "GARBAGE": 0,
        "MELTDOWN": 0,
        "ERROR": 0,
    }
    start_file_idx = 0
    start_row_idx = 0

    if args.resume:
        state = load_checkpoint(args.resume)
        if state:
            results_list = state.get("results", [])
            task_hit_l3 = state.get("task_hit_l3", 0)
            task_hit_l2 = state.get("task_hit_l2", 0)
            status_counts = state.get("status_counts", status_counts)
            start_file_idx = state.get("file_index", 0)
            start_row_idx = state.get("last_index", 0) + 1
            checkpoint_path = args.resume
            # 恢复时已打印提示

    # 如果未恢复断点，创建新的断点文件
    if not checkpoint_path:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        ckpt_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ckpt_{ckpt_timestamp}.json")
        print(f"  💾 断点文件将保存至: {checkpoint_path}")

    # 5. 遍历所有测试文件
    total_processed = 0
    t0 = time.time()

    for file_idx in range(start_file_idx, len(test_files)):
        test_csv_path = test_files[file_idx]
        if not os.path.exists(test_csv_path):
            print(f"\n⚠️ 文件不存在，跳过: {test_csv_path}")
            continue

        print(f"\n{'='*60}")
        print(f"📊 正在读取测试文件 [{file_idx+1}/{len(test_files)}]: {os.path.basename(test_csv_path)}")
        df_test = pd.read_csv(test_csv_path, encoding='utf-8-sig')
        total_records = len(df_test)
        print(f"   本文件共 {total_records} 条数据")

        # 确定起始行
        if file_idx == start_file_idx:
            row_start = start_row_idx
        else:
            row_start = 0

        for index in range(row_start, total_records):
            row = df_test.iloc[index]

            results_list, status_counts, task_hit_l3, task_hit_l2 = process_row(
                index, row, agent, retriever, results_list, status_counts, task_hit_l3, task_hit_l2
            )
            total_processed += 1

            # 检查是否达到 max_records 限制
            if args.max_records and total_processed >= args.max_records:
                print(f"\n⚠️ 已达到 --max-records={args.max_records} 限制，停止处理。")
                break

            # 断点保存
            if (index + 1) % args.checkpoint_interval == 0:
                ckpt_state = {
                    "last_index": index,
                    "file_index": file_idx,
                    "results": results_list,
                    "task_hit_l3": task_hit_l3,
                    "task_hit_l2": task_hit_l2,
                    "status_counts": status_counts,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                save_checkpoint(checkpoint_path, ckpt_state)

        if args.max_records and total_processed >= args.max_records:
            break

        # 文件处理完毕后，重置 start_row_idx（避免影响下个文件）
        start_row_idx = 0

    # 6. 导出测试统计报告
    if results_list:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"agent_eval_gemma4_{timestamp}.csv")

        df_results = pd.DataFrame(results_list)
        valid_records = len(results_list)
        acc_l3 = (task_hit_l3 / valid_records) * 100 if valid_records > 0 else 0
        acc_l2 = (task_hit_l2 / valid_records) * 100 if valid_records > 0 else 0

        # P0-F: 状态分布统计
        n_success = status_counts.get("SUCCESS", 0)
        n_review = status_counts.get("REVIEW_SUGGESTED", 0)
        n_garbage = status_counts.get("GARBAGE", 0)
        n_meltdown = status_counts.get("MELTDOWN", 0)
        n_error = status_counts.get("ERROR", 0)

        summary_row = pd.DataFrame([{
            "数据ID": "【最终统计】",
            "原始名称": f"总测试量: {valid_records} 条",
            "Top-1命中(三级)": f"{acc_l3:.2f}%",
            "Top-1命中(二级)": f"{acc_l2:.2f}%",
            "状态机终局": (
                f"SUCCESS={n_success} | REVIEW_SUGGESTED={n_review} | "
                f"GARBAGE={n_garbage} | MELTDOWN={n_meltdown} | ERROR={n_error}"
            ),
            "置信度评分": "见各行"
        }])

        df_final = pd.concat([df_results, summary_row], ignore_index=True)
        df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

        print("\n" + "="*60)
        print(f"🏆 测试跑批完毕！结果已保存至: {output_file}")
        print(f"   总耗时: {time.time() - t0:.1f}s")
        print(f"  ⭐ 【三级细分】Top-1 命中率: {acc_l3:.2f}% ({task_hit_l3}/{valid_records})")
        print(f"  ⭐⭐ 【二级大类】Top-1 命中率: {acc_l2:.2f}% ({task_hit_l2}/{valid_records})")
        print(f"  📊 状态分布:")
        print(f"     ✅ SUCCESS        (自动入库):  {n_success} 条 ({n_success/valid_records*100:.1f}%)")
        print(f"     🔍 REVIEW_SUGGESTED (人工复核): {n_review} 条 ({n_review/valid_records*100:.1f}%)")
        print(f"     🗑️  GARBAGE         (垃圾岗位):  {n_garbage} 条 ({n_garbage/valid_records*100:.1f}%)")
        print(f"     💥 MELTDOWN        (推理失败):  {n_meltdown} 条 ({n_meltdown/valid_records*100:.1f}%)")
        print(f"     ❌ ERROR           (系统异常):  {n_error} 条 ({n_error/valid_records*100:.1f}%)")
        print("="*60)

        # 清理断点文件（跑批完成后删除）
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"  🗑️ 断点文件已清理: {checkpoint_path}")
    else:
        print("\n⚠️ 无有效结果，未生成报告。")


if __name__ == "__main__":
    main()
