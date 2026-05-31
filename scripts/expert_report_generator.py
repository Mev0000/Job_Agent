import pandas as pd
import ast

def safe_literal_eval(val):
    """安全解析列表字符串"""
    if pd.isna(val) or val == '[]':
        return ""
    try:
        items = ast.literal_eval(str(val))
        if isinstance(items, list):
            # 将列表转为逗号分隔的字符串，方便人类阅读
            return "、".join(items)
    except:
        pass
    return str(val)

def main():
    print("="*50)
    print(" 📊 正在生成《职业大典防混淆专家审查报告》...")
    print("="*50)
    
    # 1. 读取基础数据 (请确保你已经跑过了之前的清洗脚本，如果没有，就读取未清洗的版本)
    try:
        df_nodes = pd.read_csv('Nodes_Cleaned.csv', dtype={'职业编码': str})
    except FileNotFoundError:
        print("⚠️ 未找到 Nodes_Cleaned.csv，降级使用未清洗的 Nodes.csv")
        df_nodes = pd.read_csv('Nodes.csv', dtype={'职业编码': str})
        
    df_edges = pd.read_csv('Edges_Confused_Final.csv', dtype={'源编码': str, '目标编码': str})
    
    # 构建快速查询字典，用空间换时间
    nodes_dict = df_nodes.set_index('职业编码').to_dict('index')
    
    report_data = []
    
    # 2. 遍历每一条混淆边，拉取双方的原子要素
    for idx, row in df_edges.iterrows():
        code_A = str(row['源编码'])
        code_B = str(row['目标编码'])
        
        node_A = nodes_dict.get(code_A, {})
        node_B = nodes_dict.get(code_B, {})
        
        report_row = {
            '岗位A_编码': code_A,
            '岗位A_名称': row['源名称'],
            '岗位B_编码': code_B,
            '岗位B_名称': row['目标名称'],
            
            '🤖 LLM 核心辨析口诀 (Description > Name)': row.get('LLM防坑口诀', ''),
            '🧲 向量相似度': row.get('相似度得分', ''),
            
            # 岗位 A 的原子要素
            '岗位A_核心动作': safe_literal_eval(node_A.get('Extracted_Actions', '')),
            '岗位A_作用对象': safe_literal_eval(node_A.get('Extracted_Objects', '')),
            '岗位A_工作环境': safe_literal_eval(node_A.get('Extracted_Environments', '')),
            
            # 岗位 B 的原子要素
            '岗位B_核心动作': safe_literal_eval(node_B.get('Extracted_Actions', '')),
            '岗位B_作用对象': safe_literal_eval(node_B.get('Extracted_Objects', '')),
            '岗位B_工作环境': safe_literal_eval(node_B.get('Extracted_Environments', '')),
            
            # 原始文本，供专家兜底核对
            '岗位A_法定描述': str(node_A.get('职业描述', '')) + " " + str(node_A.get('主要工作任务', '')),
            '岗位B_法定描述': str(node_B.get('职业描述', '')) + " " + str(node_B.get('主要工作任务', ''))
        }
        report_data.append(report_row)
        
    df_report = pd.DataFrame(report_data)
    
    # 3. 导出为高可读性的 Excel
    output_excel = '职业大典易混淆岗位_专家审查版.xlsx'
    
    # 使用 xlsxwriter 引擎，方便进行格式控制
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        df_report.to_excel(writer, index=False, sheet_name='专家审查视图')
        workbook = writer.book
        worksheet = writer.sheets['专家审查视图']
        
        # 定义格式
        header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'top', 
            'fg_color': '#D7E4BC', 'border': 1
        })
        text_wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
        
        # 写入表头并设置格式
        for col_num, value in enumerate(df_report.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        # 设置列宽，让专家看着舒服
        worksheet.set_column('A:D', 15, text_wrap_format)
        worksheet.set_column('E:E', 40, text_wrap_format) # 口诀列给宽一点
        worksheet.set_column('F:F', 12, text_wrap_format)
        worksheet.set_column('G:L', 20, text_wrap_format) # 要素列
        worksheet.set_column('M:N', 50, text_wrap_format) # 描述列最宽
        
        # 冻结首行和前两列
        worksheet.freeze_panes(1, 4)

    print(f"🎉 专家审查报告生成完毕！")
    print(f"📁 已保存至: {output_excel}")
    print("💡 专家打开 Excel 时：首行已冻结，原子要素已合并，列宽已自适应。")

if __name__ == "__main__":
    main()