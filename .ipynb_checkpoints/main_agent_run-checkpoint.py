import os
import time
import datetime
import pandas as pd
import re
import yaml
import json
import torch

# 导入核心模块
from core.state_machine import JobAgentStateMachine
from core.retriever import AdvancedRetriever

# ==========================================
# 1. 路径与配置设定
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw_jd", "2025_5.18_100.csv")
DICT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw_jd", "2022年职业分类大典（整体修订）.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "logs")

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
    if not os.path.exists(DICT_CSV_PATH):
        raise FileNotFoundError(f"找不到职业大典源文件: {DICT_CSV_PATH}")
        
    df = pd.read_csv(DICT_CSV_PATH).fillna("")
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
                # 必须用明确的标识符，方便 retriever 和 Hop-2 用正则切除
                text += f" 主要工作任务：{tasks}"
            
            corpus.append({
                "code": code,
                "name": name,
                "text": text
            })
            
    print(f"✅ 向量语料库构建完毕，共提取 {len(corpus)} 条记录 (已挂载全量工作任务)。")
    return corpus

def main():
    print("="*60)
    print("🚀 Job Agent (Dual-Track RAG Edition) - 批量测试框架启动中...")
    print("="*60)
    
    # 1. 加载配置与动态语料
    config = load_config()
    occupation_corpus = load_occupation_corpus()
    
    # 2. 初始化核心组件
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
    
    # 3. 读取测试数据集
    if not os.path.exists(TEST_CSV_PATH):
        print(f"❌ 找不到测试集文件: {TEST_CSV_PATH}")
        return
        
    df_test = pd.read_csv(TEST_CSV_PATH, encoding='utf-8-sig')
    total_records = len(df_test)
    print(f"✅ 测试集加载完毕，共 {total_records} 条数据准备测试。\n")
    print("-" * 60)

    results_list = []
    task_hit_l3 = 0  
    task_hit_l2 = 0  
    
    # 4. 开始跑批循环
    for index, row in df_test.iterrows():
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
            continue
            
        truth_occ_code = ", ".join(list(acceptable_truths))
        acceptable_truths_lvl2 = {"-".join(c.split('-')[:2]) for c in acceptable_truths if "-" in c}
        
        print(f"\n▶️ [{index+1}/{total_records}] 正在处理: 【{raw_name}】")
        start_t = time.time()
        
        try:
            print("  🔍 正在通过双轨引擎进行全域扫描与图谱安检...")
            initial_context = retriever.retrieve(raw_name, raw_desc)

            torch.cuda.empty_cache()
            
            result_dict = agent.run(raw_name, raw_desc, initial_context)
            final_state = result_dict.get("status", "ERROR")
            predicted_code = result_dict.get("code", "未提取")
            reasoning_log = json.dumps(result_dict.get("reasoning", {}), ensure_ascii=False)
            
        except Exception as e:
            print(f"  ❌ 处理异常: {e}")
            final_state = "ERROR"
            predicted_code = "未提取"
            reasoning_log = f"运行报错: {str(e)}"
        
        # 计算命中率 (兼容大模型越级预测到四级细分的情况)
        if predicted_code not in ["未提取", "ERROR", "MANUAL_REVIEW", "8-00-00"]:
            parts = predicted_code.split('-')
            # 智能截断：如果预测了 2-02-10-03，强行截断为 2-02-10 用于比对三级标签
            predicted_lvl3 = "-".join(parts[:3]) if len(parts) >= 3 else predicted_code
            predicted_lvl2 = "-".join(parts[:2]) if len(parts) >= 2 else predicted_code
            
            is_hit_l3 = predicted_lvl3 in acceptable_truths
            is_hit_l2 = predicted_lvl2 in acceptable_truths_lvl2
        else:
            is_hit_l3 = False
            is_hit_l2 = False

        if is_hit_l3: task_hit_l3 += 1
        if is_hit_l2: task_hit_l2 += 1

        print(f"  [耗时: {time.time() - start_t:.1f}s]")
        print(f"  🏆 真实标签池 (三级): {truth_occ_code}")
        print(f"  🤖 预测结果: {predicted_code} (状态: {final_state})")
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
            "Agent推理日志": reasoning_log
        })

    # 5. 导出测试统计报告
    if results_list:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"agent_eval_gemma4_{timestamp}.csv")
        
        df_results = pd.DataFrame(results_list)
        valid_records = len(results_list)
        acc_l3 = (task_hit_l3 / valid_records) * 100
        acc_l2 = (task_hit_l2 / valid_records) * 100
        
        summary_row = pd.DataFrame([{
            "数据ID": "【最终统计】", 
            "原始名称": f"总测试量: {valid_records} 条",
            "Top-1命中(三级)": f"{acc_l3:.2f}%",
            "Top-1命中(二级)": f"{acc_l2:.2f}%"
        }])
        
        df_final = pd.concat([df_results, summary_row], ignore_index=True)
        df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*60)
        print(f"🏆 测试跑批完毕！结果已保存至: {output_file}")
        print(f"  ⭐ 【三级细分】Top-1 命中率: {acc_l3:.2f}% ({task_hit_l3}/{valid_records})")
        print(f"  ⭐⭐ 【二级大类】Top-1 命中率: {acc_l2:.2f}% ({task_hit_l2}/{valid_records})")
        print("="*60)

if __name__ == "__main__":
    main()