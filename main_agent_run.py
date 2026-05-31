import os
import time
import pickle
import datetime
import pandas as pd
import re

# 导入你现有的核心模块
from core.state_machine import JobAgentStateMachine
from core.retriever import AdvancedRetriever # 假设你之前的检索器是这个名字，如果是别的请修改
import yaml # 用于简单加载 config

# ==========================================
# 1. 路径与配置设定 (完全适配当前目录)
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# 假设测试集也放在 raw_jd 下
TEST_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw_jd", "2025_5.18_100.csv")
DICT_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "cache", "global_dict_tree.pkl")
# 输出结果放在 logs 目录下
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output")

def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def load_global_dict_cache():
    if not os.path.exists(DICT_CACHE_PATH):
        raise FileNotFoundError(f"找不到字典缓存，请先运行 build_dict_cache.py: {DICT_CACHE_PATH}")
    print("⏳ 正在极速加载全局大典缓存...")
    with open(DICT_CACHE_PATH, "rb") as f:
        tree = pickle.load(f)
    print("✅ 大典缓存加载完毕！")
    return tree

def main():
    print("="*60)
    print("🚀 Job Agent (Gemma 4 Edition) - 批量测试框架启动中...")
    print("="*60)
    
    # 1. 加载配置和大典缓存
    config = load_config()
    global_dict_tree = load_global_dict_cache()
    
    # 2. 初始化核心组件 (保持你原有的初始化方式)
    print("\n[模块加载 1/2] 初始化检索引擎...")
    retriever = AdvancedRetriever(config) # 使用你现在的检索器初始化方式
    
    print("[模块加载 2/2] 初始化 Agent 状态机...")
    agent = JobAgentStateMachine(config, retriever)
    
    # 3. 读取测试数据集
    if not os.path.exists(TEST_CSV_PATH):
        print(f"❌ 找不到测试集文件: {TEST_CSV_PATH}")
        return
        
    df_test = pd.read_csv(TEST_CSV_PATH, encoding='utf-8-sig')
    total_records = len(df_test)
    print(f"✅ 测试集加载完毕，共 {total_records} 条数据准备测试。")
    print("-" * 60)

    # 4. 统计指标计数器
    results_list = []
    task_hit_l3 = 0  
    task_hit_l2 = 0  
    
    # 5. 开始跑批循环
    for index, row in df_test.iterrows():
        row_id = row.get('_id', f"IDX_{index}")
        raw_name = str(row.get('job_name', '')).strip()
        raw_desc = str(row.get('job_descrip', '')).strip()
        
        # 提取 Ground Truth
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
        
        # ==========================================
        # 核心调用：触发 Agent 状态机
        # ==========================================
        # 你的 agent.run 需要返回最终结果字典或直接返回最终代码
        try:
            # 根据你之前设计的状态机，这里调用 run 并获取最后确定的分类代码
            final_result = agent.run(raw_name, raw_desc, "初始化上下文(可置空或用大典宏观结构)")
            
            # 解析状态机结果 (这里可能需要根据你 agent 的实际返回格式微调)
            if isinstance(final_result, dict):
                predicted_code = final_result.get("result_code", "未提取")
                final_state = final_result.get("status", "UNKNOWN")
            else:
                predicted_code = final_result if final_result else "未提取"
                final_state = "SUCCESS" if final_result else "FAILED"
                
            reasoning_log = "详见终端输出" # 如果你能从 agent 拿到完整思考流最好，否则先简单记录
            
        except Exception as e:
            print(f"❌ 处理异常: {e}")
            predicted_code = "处理报错"
            final_state = "ERROR"
            reasoning_log = str(e)
        
        # 计算命中率
        predicted_lvl2 = "-".join(predicted_code.split('-')[:2]) if "-" in predicted_code else "未提取"

        is_hit_l3 = (predicted_code in acceptable_truths) if predicted_code != "未提取" else False
        is_hit_l2 = (predicted_lvl2 in acceptable_truths_lvl2) if predicted_lvl2 != "未提取" else False

        if is_hit_l3: task_hit_l3 += 1
        if is_hit_l2: task_hit_l2 += 1

        print(f"  [耗时: {time.time() - start_t:.1f}s]")
        print(f"  🏆 真实标签池 (三级): {truth_occ_code}")
        print(f"  🤖 预测结果: {predicted_code} (状态: {final_state})")
        print(f"  🏅 三级命中: {'✅' if is_hit_l3 else '❌'}  |  二级命中: {'✅' if is_hit_l2 else '❌'}")
        
        # 记录结果
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

    # 6. 导出报告
    if results_list:
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