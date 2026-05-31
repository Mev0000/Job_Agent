# build_dict_cache.py (可以放在项目根目录，也可以建个 scripts 文件夹放进去)
import os
import re
import pandas as pd
import pickle

def build_and_save_dict_tree():
    # 使用你真实的目录结构
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
    # 假设你把大典放在了 data/raw_jd/ 下
    DICT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw_jd", "2022年职业分类大典（整体修订）.csv")
    # 缓存放到现有的 data/cache/ 目录下
    CACHE_SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "cache", "global_dict_tree.pkl")

    print(f"⏳ 正在从 {DICT_CSV_PATH} 读取并解析职业大典...")
    
    if not os.path.exists(DICT_CSV_PATH):
        print(f"❌ 错误：找不到大典文件，请确认是否放在了 {DICT_CSV_PATH}")
        return

    try:
        df_dict_full = pd.read_csv(DICT_CSV_PATH).fillna("")
        GLOBAL_DICT_TREE = {}
        
        for _, row in df_dict_full.iterrows():
            code = str(row['职业编码']).strip()
            dash_count = code.count('-')
            
            if dash_count not in [2, 3]: 
                continue
                
            level2_prefix = "-".join(code.split('-')[:2])
            if level2_prefix not in GLOBAL_DICT_TREE:
                GLOBAL_DICT_TREE[level2_prefix] = {}
                
            name = str(row['职业名称']).strip()
            desc = re.sub(r'\s+', ' ', str(row['职业描述']).strip())
            
            if dash_count == 2: 
                l3_code = code
                if l3_code not in GLOBAL_DICT_TREE[level2_prefix]:
                    GLOBAL_DICT_TREE[level2_prefix][l3_code] = {"name": name, "desc": desc, "l4_list": []}
                else:
                    GLOBAL_DICT_TREE[level2_prefix][l3_code]["name"] = name
                    GLOBAL_DICT_TREE[level2_prefix][l3_code]["desc"] = desc
                    
            elif dash_count == 3: 
                l3_parent_code = "-".join(code.split('-')[:3])
                if l3_parent_code not in GLOBAL_DICT_TREE[level2_prefix]:
                    GLOBAL_DICT_TREE[level2_prefix][l3_parent_code] = {"name": "未知三级分类", "desc": "", "l4_list": []}
                
                l4_feature = f"【细分四级】{name}：{desc}"
                GLOBAL_DICT_TREE[level2_prefix][l3_parent_code]["l4_list"].append(l4_feature)

        # 序列化保存
        with open(CACHE_SAVE_PATH, 'wb') as f:
            pickle.dump(GLOBAL_DICT_TREE, f)
            
        print(f"✅ 全局字典缓存构建成功！共加载 {len(GLOBAL_DICT_TREE)} 个二级类目。")
        print(f"📁 缓存文件已保存至: {CACHE_SAVE_PATH}")
        
    except Exception as e:
        print(f"⚠️ 解析失败: {e}")

if __name__ == "__main__":
    build_and_save_dict_tree()