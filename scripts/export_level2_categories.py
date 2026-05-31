import pandas as pd
import os

# ================= 配置区 =================
CSV_FILE_PATH = '2022年职业分类大典（整体修订）.csv'
OUTPUT_TXT_PATH = '二级大类清单_提供给LLM分析使用.txt'
OUTPUT_CSV_PATH = '二级大类_结构化备份.csv'
# ==========================================

def main():
    print(f"📂 正在读取源文件: {CSV_FILE_PATH}")
    if not os.path.exists(CSV_FILE_PATH):
        print("❌ 错误：找不到源 CSV 文件，请检查路径！")
        return

    df = pd.read_csv(CSV_FILE_PATH)

    # 1. 精准过滤：只保留“二级”中类
    df_level2 = df[df['层级'] == '二级'].copy()
    
    # 2. 提取核心列并处理缺失值
    # 二级大类通常没有“主要工作任务”，所以只需提取编码、名称和描述
    df_level2['职业描述'] = df_level2['职业描述'].fillna('无官方描述补充')
    
    # 替换掉描述中的换行符，防止导出文本格式错乱
    df_level2['职业描述'] = df_level2['职业描述'].str.replace('\n', '', regex=False).str.replace('\r', '', regex=False)

    # 3. 结构化保存 (供本地 Excel 查阅)
    df_level2[['职业编码', '职业名称', '职业描述']].to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    
    # 4. 生成给 LLM 阅读的纯文本格式 (这是最关键的一步)
    print(f"📝 正在生成提供给 LLM 裁判的文本清单...")
    with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as f:
        f.write("《2022年职业分类大典》二级中类名录：\n")
        f.write("="*50 + "\n")
        for index, row in df_level2.iterrows():
            line = f"[{row['职业编码']}] {row['职业名称']} - 官方定义：{row['职业描述']}\n"
            f.write(line)
            
    print(f"✅ 提取成功！共提取了 {len(df_level2)} 个二级大类。")
    print(f"   -> 文本清单已保存至: {OUTPUT_TXT_PATH} (你可以直接把里面的内容复制给大模型)")
    print(f"   -> 表格备份已保存至: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()