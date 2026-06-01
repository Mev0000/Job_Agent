import pandas as pd
import networkx as nx
import os
import pickle
import ast

class DictGraphRAG:
    def __init__(self, data_dir="data/"):
        """
        初始化大典职业图谱构建器
        :param data_dir: 存放 5 张核心 CSV 表格的文件夹路径
        """
        self.data_dir = data_dir
        self.G = nx.DiGraph() # 初始化有向图
        
    def _safe_literal_eval(self, val):
        """安全地将字符串形态的列表（如 "['动作1', '动作2']"）转为 Python List"""
        if pd.isna(val) or val == "":
            return []
        try:
            return ast.literal_eval(val)
        except:
            return [str(val)]

    def build_graph(self):
        """从 CSV 文件全量构建图谱"""
        print("🚀 开始构建大典知识图谱...")
        
        # 1. 载入核心节点与属性 (Nodes_Cleaned.csv)
        nodes_df = pd.read_csv(os.path.join(self.data_dir, "Nodes_Cleaned.csv"), dtype=str).fillna("")
        for _, row in nodes_df.iterrows():
            code = row['职业编码'].strip()
            self.G.add_node(
                code,
                node_type="Job",
                level=row['层级'],
                name=row['职业名称'],
                desc=row['职业描述'],
                parent_code=row['Parent_Code'],
                l2_prefix=row['L2_Prefix'],
                # 法理合规布尔值
                is_gov=row['Is_Government'] == 'True',
                is_med=row['Is_Medical_Clinical'] == 'True',
                is_prod=row['Is_Production'] == 'True',
                # 原子要素 (使用 AST 将字符串转回列表)
                actions=self._safe_literal_eval(row['Extracted_Actions']),
                objects=self._safe_literal_eval(row['Extracted_Objects']),
                envs=self._safe_literal_eval(row['Extracted_Environments'])
            )
        print(f"✅ 载入职业节点: {len(nodes_df)} 个")

        # 2. 载入层级从属边 (Edges_Taxonomy.csv)
        tax_df = pd.read_csv(os.path.join(self.data_dir, "Edges_Taxonomy.csv"), dtype=str).fillna("")
        for _, row in tax_df.iterrows():
            self.G.add_edge(row['Source'], row['Target'], relation=row['Relation'])
            
        # 3. 载入实体特征边 (Edges_Entities_Cleaned.csv) - 可选，用于多跳下钻
        ent_df = pd.read_csv(os.path.join(self.data_dir, "Edges_Entities_Cleaned.csv"), dtype=str).fillna("")
        for _, row in ent_df.iterrows():
            # 实体节点可能还不存在，安全起见先添加
            self.G.add_node(row['目标实体'], node_type="Entity")
            self.G.add_edge(row['源编码'], row['目标实体'], relation=row['关系类型'])

        # 4. 载入致命法理红线边 (Edges_Wormhole.csv)
        wormhole_df = pd.read_csv(os.path.join(self.data_dir, "Edges_Wormhole.csv"), dtype=str).fillna("")
        for _, row in wormhole_df.iterrows():
            self.G.add_edge(
                row['源编码'], 
                row['目标编码'], 
                relation=row['关系类型'], 
                rule_desc=row['规则说明']
            )
        print(f"✅ 载入法理红线边 (Wormholes): {len(wormhole_df)} 条")

        # 5. 载入易混淆口诀边 (Edges_Confused_Final.csv)
        confuse_df = pd.read_csv(os.path.join(self.data_dir, "Edges_Confused_Final.csv"), dtype=str).fillna("")
        for _, row in confuse_df.iterrows():
            self.G.add_edge(
                row['源编码'], 
                row['目标编码'], 
                relation=row['关系类型'], 
                rule_desc=row['LLM防坑口诀']
            )
        print(f"✅ 载入易混淆口诀边: {len(confuse_df)} 条")
        print(f"🎉 图谱构建完成！总节点数: {self.G.number_of_nodes()}, 总边数: {self.G.number_of_edges()}")

    def save_to_disk(self, filepath="graph_rag.pkl"):
        """将图谱序列化保存，下次秒级加载"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.G, f)
        print(f"💾 图谱已缓存至 {filepath}")

    def load_from_disk(self, filepath="graph_rag.pkl"):
        """读取缓存的图谱"""
        with open(filepath, 'rb') as f:
            self.G = pickle.load(f)
        print(f"🚀 图谱极速加载完毕！")

    # ================= 供大模型 (Agent) 调用的三大核心接口 =================

    def get_job_features(self, code):
        """接口 1：获取职业的原子特征和红线标识"""
        if not self.G.has_node(code):
            return None
        data = self.G.nodes[code]
        if data.get("node_type") != "Job":
            return None
            
        return {
            "名称": data.get("name", ""),
            "动作": data.get("actions", []),
            "对象": data.get("objects", []),
            "环境": data.get("envs", []),
            "是否涉公权": data.get("is_gov", False),
            "是否涉临床": data.get("is_med", False)
        }

    def check_wormhole(self, code_a, code_b):
        """接口 2：检查两个代码之间是否存在绝对红线 (MUTUALLY_EXCLUSIVE)"""
        # 图是无向感知的红线，需双向检查
        if self.G.has_edge(code_a, code_b) and self.G.edges[code_a, code_b].get("relation") == "MUTUALLY_EXCLUSIVE":
            return self.G.edges[code_a, code_b].get("rule_desc")
        if self.G.has_edge(code_b, code_a) and self.G.edges[code_b, code_a].get("relation") == "MUTUALLY_EXCLUSIVE":
            return self.G.edges[code_b, code_a].get("rule_desc")
        return None

    def get_confusion_rule(self, code_a, code_b):
        """接口 3：提取防坑口诀 (POTENTIALLY_CONFUSED)"""
        if self.G.has_edge(code_a, code_b) and self.G.edges[code_a, code_b].get("relation") == "POTENTIALLY_CONFUSED":
            return self.G.edges[code_a, code_b].get("rule_desc")
        if self.G.has_edge(code_b, code_a) and self.G.edges[code_b, code_a].get("relation") == "POTENTIALLY_CONFUSED":
            return self.G.edges[code_b, code_a].get("rule_desc")
        return None