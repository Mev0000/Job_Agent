import pandas as pd
import ast
import re

# ==========================================
# 🛑 第一级：专家核心黑名单 (精确匹配)
# 来源：专家核对版《职业大典无用关键词_0602.docx》
# ==========================================

# 动词类黑名单
ACTION_STOP_WORDS = {
    '从事', '进行', '处理', '提供', '实施', '参与', '负责', '开展', '组织', '办理', '服务',
    '使用', '协助', '执行', '利用', '选择', '完成', '制定', '确定', '提出', '担任', '建立',
    '提供服务', '组织实施', '协调', '发生', '运用', '相关', '其他', '具有',
    '履行', '履行职责', '行使', '沟通', '评论', '相互作用', '聆听'
}

# 实体/对象/环境类黑名单
ENTITY_STOP_WORDS = {
    # 抽象与泛指事务
    '工作', '业务', '事项', '事务', '项目', '计划', '目标', '任务', '日常工作', '行政事务',
    '相关', '其他', '人员', '问题', '活动', '领导职务', '职权', '情况', '机构', '单位', 
    '有关', '及其他', '其中', '有关规定', '其他职责', '等工作', '会议', '工作机构', '会员', 
    '工作报告', '相关机构', '指挥单位', '总体规划', '企业', '工作计划',
    # 泛指设备与系统
    '设备', '专用设备', '生产设备', '成型设备', '加工设备', '配料设备', '成套设备', '附属设备', 
    '附属设施', '环保设施', '计量仪器', '消防救生设备', '设备工装', '仪器仪表', '工艺装备', '辅助装置',
    '系统', '远程设备控制系统', '机具', '机台', '生产系统',
    # 泛指数据与参数
    '生产数据', '生产记录', '文档', '资料', '功能模块', '设备运行状态', '故障', '运行参数', '工艺参数',
    # 泛指原料与物料
    '原料', '原材料', '原料和产品', '配料', '原辅料', '成品', '构件', '物料', '零部件', '零件', 
    '配件', '外购件', '耗材', '部件', '整机', '零件和部件', '总成', '结构件', '物体', '洒落物',
    # 泛指环境与场所
    '办公室', '生产线', '巷道', '工作面', '地面', '外场', '装饰', '作业场地', '工作场地', 
    '现场', '迹地', '集材道', '生产环境', '评吸场所', '作业区', '生产场地'
}

# ==========================================
# 🛑 第二级：垃圾外壳正则 (用于剥壳检测)
# ==========================================
GARBAGE_PREFIXES = r'^(相关|其他|各种|各类|有关|某些|一般|日常的?|简单的?)'
GARBAGE_SUFFIXES = r'(等工作|等相关工作|及其他|等事项|等相关事宜|等)$'

def is_meaningless(entity_str, stop_words_set):
    """三级漏斗深度清洗逻辑"""
    if not entity_str: return True
    
    clean_str = str(entity_str).strip()
    
    # 1. 精确命中黑名单
    if clean_str in stop_words_set: return True
    
    # 2. 剥去垃圾前后缀后，再看核心词是否在黑名单中 (如 "相关设备" -> 命中 "设备")
    core_str = re.sub(GARBAGE_PREFIXES, '', clean_str)
    core_str = re.sub(GARBAGE_SUFFIXES, '', core_str).strip()
    if core_str in stop_words_set or not core_str: return True
        
    # 3. 拦截大模型提取的敷衍长句 (例如 "进行相关工作", "负责其他业务")
    if len(clean_str) > 4 and ('相关工作' in clean_str or '其他业务' in clean_str or '有关规定' in clean_str):
        return True

    return False

def clean_nodes(input_path='Nodes.csv', output_path='Nodes_Cleaned.csv'):
    print(f"🧹 开始深度清洗基础节点表 ({input_path})...")
    try:
        # 强制指定职业编码为字符串，防止前导0丢失
        df_nodes = pd.read_csv(input_path, dtype={'职业编码': str})
        
        def filter_list_string(list_str, stop_words_set):
            """安全解析列表字符串 (如 "['装配', '调试']") 并进行深度过滤"""
            if pd.isna(list_str): return "[]"
            try:
                # 安全评估字符串形态的列表
                items = ast.literal_eval(str(list_str))
                if isinstance(items, list):
                    # 遍历列表，过滤掉废话，保留高价值实体
                    cleaned = [item for item in items if not is_meaningless(item, stop_words_set)]
                    return str(cleaned) # 重新转回字符串存储
            except Exception:
                pass
            return str(list_str)

        # 针对上传的 Nodes_Cleaned.csv 中的三大要素列进行精准清洗
        if 'Extracted_Actions' in df_nodes.columns:
            df_nodes['Extracted_Actions'] = df_nodes['Extracted_Actions'].apply(
                lambda x: filter_list_string(x, ACTION_STOP_WORDS)
            )
        if 'Extracted_Objects' in df_nodes.columns:
            df_nodes['Extracted_Objects'] = df_nodes['Extracted_Objects'].apply(
                lambda x: filter_list_string(x, ENTITY_STOP_WORDS)
            )
        if 'Extracted_Environments' in df_nodes.columns:
            df_nodes['Extracted_Environments'] = df_nodes['Extracted_Environments'].apply(
                lambda x: filter_list_string(x, ENTITY_STOP_WORDS)
            )
            
        df_nodes.to_csv(output_path, index=False, encoding='utf-8-sig')
        print("   -> 节点表要素清洗完毕！所有元数据特征均已无损保留。")
    except Exception as e:
        print(f"   ❌ 节点表清洗出错或文件不存在: {e}")

if __name__ == "__main__":
    # 请确保同目录下存在未经过滤的原始 Nodes.csv 和 Edges_Entities.csv
    # 如果只有之前的 Nodes_Cleaned.csv，也可以把它重命名为 Nodes.csv 作为输入
    clean_nodes(input_path='Nodes.csv', output_path='Nodes_Cleaned.csv')