import pandas as pd
import ast

# ==========================================
# 🛑 定义黑名单 (基于真实数据频次提炼)
# ==========================================
ACTION_STOP_WORDS = {
    '从事', '进行', '处理', '提供', '实施', '参与', '负责', '开展', '组织', '办理', '服务',
    '使用', '协助', '执行', '利用', '选择', '完成', '制定', '确定', '提出', '担任', '建立',
    '提供服务', '组织实施', '协调', '发生', '运用', '相关', '其他', '具有'
}

OBJECT_STOP_WORDS = {
    '工作', '业务', '事项', '事务', '项目', '计划', '目标', '任务', '日常工作', '行政事务',
    '相关', '其他', '人员', '问题', '活动', '领导职务', '职权'
}

def clean_edges():
    print("🧹 开始清洗实体边表 (Edges_Entities.csv)...")
    df_edges = pd.read_csv('Edges_Entities.csv')
    initial_len = len(df_edges)
    
    def should_keep(row):
        entity = str(row['目标实体']).strip()
        rel_type = row['关系类型']
        if rel_type == 'INVOLVES_ACTION' and entity in ACTION_STOP_WORDS: return False
        if rel_type == 'TARGETS_OBJECT' and entity in OBJECT_STOP_WORDS: return False
        return True
        
    df_clean = df_edges[df_edges.apply(should_keep, axis=1)]
    df_clean.to_csv('Edges_Entities_Cleaned.csv', index=False, encoding='utf-8-sig')
    print(f"   -> 边表清洗完毕！清理了 {initial_len - len(df_clean)} 条无意义关系。")

def clean_nodes():
    print("🧹 开始清洗基础节点表 (Nodes.csv)...")
    df_nodes = pd.read_csv('Nodes.csv', dtype={'职业编码': str})
    
    def remove_stop_words_from_list_str(list_str, stop_words):
        """安全地解析字符串列表，剔除黑名单词汇，再转回字符串"""
        if pd.isna(list_str): return "[]"
        try:
            items = ast.literal_eval(str(list_str))
            if isinstance(items, list):
                cleaned = [item for item in items if item not in stop_words]
                return str(cleaned) # 转回字符串存储
        except Exception:
            pass
        return str(list_str)

    # 清洗动词列
    df_nodes['Extracted_Actions'] = df_nodes['Extracted_Actions'].apply(
        lambda x: remove_stop_words_from_list_str(x, ACTION_STOP_WORDS)
    )
    # 清洗对象列
    df_nodes['Extracted_Objects'] = df_nodes['Extracted_Objects'].apply(
        lambda x: remove_stop_words_from_list_str(x, OBJECT_STOP_WORDS)
    )
    
    df_nodes.to_csv('Nodes_Cleaned.csv', index=False, encoding='utf-8-sig')
    print("   -> 节点表清洗完毕！节点内部属性中的脏词已抹除。")

if __name__ == "__main__":
    print("="*50)
    print(" 🚿 启动 GraphRAG 无意义词汇物理清洗程序")
    print("="*50)
    clean_edges()
    clean_nodes()
    print("\n🎉 清洗全部完成！请在后续图谱入库时使用带 '_Cleaned' 后缀的文件。")