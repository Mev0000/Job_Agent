import pandas as pd
import networkx as nx
import os
import pickle
import ast
from typing import List, Dict, Optional, Any

class DictGraphRAG:
    def __init__(self, data_dir: str = "data/graph_tables/"):
        """
        初始化大典职业图谱构建器
        :param data_dir: 存放核心 CSV 表格的文件夹路径
        """
        self.data_dir = data_dir
        # 主体保持有向图 (DiGraph)，因为层级 Taxonomy 和要素实体是单向从属的
        self.G = nx.DiGraph() 
        
    def _safe_literal_eval(self, val: Any) -> List[str]:
        """安全地将字符串形态的列表（如 "['动作1', '动作2']"）转为 Python List"""
        if pd.isna(val) or val == "":
            return []
        try:
            return ast.literal_eval(val)
        except:
            return [str(val)]

    def build_graph(self, seven_d_data: Dict[str, Dict] = None):
        """从 CSV 文件全量构建图谱，可选注入蒸馏7D特征
        :param seven_d_data: {code: {core_actions, objects, deliverables, ...}} 蒸馏输出
        """
        print("🚀 开始构建大典知识图谱...")
        
        # 1. 载入核心节点与属性 (Nodes_Cleaned.csv)
        nodes_path = os.path.join(self.data_dir, "Nodes_Cleaned.csv")
        if os.path.exists(nodes_path):
            nodes_df = pd.read_csv(nodes_path, dtype=str).fillna("")
            for _, row in nodes_df.iterrows():
                code = row['职业编码'].strip()
                self.G.add_node(
                    code,
                    node_type="Job",
                    level=row.get('层级', ''),
                    name=row.get('职业名称', ''),
                    desc=row.get('职业描述', ''),
                    parent_code=row.get('Parent_Code', ''),
                    l2_prefix=row.get('L2_Prefix', ''),
                    # 稳健的布尔值转换
                    is_gov=str(row.get('Is_Government', '')).lower() == 'true',
                    is_med=str(row.get('Is_Medical_Clinical', '')).lower() == 'true',
                    is_prod=str(row.get('Is_Production', '')).lower() == 'true',
                    # 原子要素
                    actions=self._safe_literal_eval(row.get('Extracted_Actions', '')),
                    objects=self._safe_literal_eval(row.get('Extracted_Objects', '')),
                    envs=self._safe_literal_eval(row.get('Extracted_Environments', ''))
                )
            print(f"✅ 载入职业节点: {len(nodes_df)} 个")

        # ── 1.5 注入蒸馏7D特征（仅四级Job节点）──
        if seven_d_data:
            enriched, skipped = 0, 0
            for code in self.G.nodes():
                node = self.G.nodes[code]
                if node.get('node_type') != 'Job':
                    continue
                if code not in seven_d_data:
                    continue
                d7 = seven_d_data[code]
                if d7.get('_quality') == 'LOW':
                    skipped += 1
                    continue
                for d7_key in ['core_actions', 'objects', 'deliverables', 'main_kpi',
                               'environment', 'served_population', 'role_level', 'category']:
                    val = d7.get(d7_key)
                    if val is not None and val != [] and val != '':
                        node[d7_key] = val
                enriched += 1
            total = len(seven_d_data)
            print(f"✅ 注入7D特征: {enriched}/{total} 个节点（跳过 {skipped} 个LOW质量）")

        # 2. 载入层级从属边 (单向边)
        tax_path = os.path.join(self.data_dir, "Edges_Taxonomy.csv")
        if os.path.exists(tax_path):
            tax_df = pd.read_csv(tax_path, dtype=str).fillna("")
            for _, row in tax_df.iterrows():
                self.G.add_edge(row['Source'], row['Target'], relation=row.get('Relation', 'is_child_of'))

        # 3. 载入实体特征边 (单向边 - 用于多跳下钻)
        ent_path = os.path.join(self.data_dir, "Edges_Entities_Cleaned.csv")
        if os.path.exists(ent_path):
            ent_df = pd.read_csv(ent_path, dtype=str).fillna("")
            for _, row in ent_df.iterrows():
                self.G.add_node(row['目标实体'], node_type="Entity")
                self.G.add_edge(row['源编码'], row['目标实体'], relation=row.get('关系类型', 'has_entity'))

        # 4. 载入致命法理红线边 (🔥重构：强制注入双向边)
        wormhole_path = os.path.join(self.data_dir, "Edges_Wormhole.csv")
        if os.path.exists(wormhole_path):
            wormhole_df = pd.read_csv(wormhole_path, dtype=str).fillna("")
            for _, row in wormhole_df.iterrows():
                src, tgt = row['源编码'], row['目标编码']
                rel, rule = row.get('关系类型', 'MUTUALLY_EXCLUSIVE'), row.get('规则说明', '')
                
                # 因为互斥是绝对对称的，我们一次性注入双向边，后续查询直接免除 if/else 冗余反查
                self.G.add_edge(src, tgt, relation=rel, rule_desc=rule)
                self.G.add_edge(tgt, src, relation=rel, rule_desc=rule)
            print(f"✅ 载入法理红线边 (Wormholes): {len(wormhole_df)} 条")

        # 5. 载入易混淆口诀边 (🔥重构：强制注入双向边)
        confuse_path = os.path.join(self.data_dir, "Edges_Confused_Final.csv")
        if os.path.exists(confuse_path):
            confuse_df = pd.read_csv(confuse_path, dtype=str).fillna("")
            for _, row in confuse_df.iterrows():
                src, tgt = row['源编码'], row['目标编码']
                rel, rule = row.get('关系类型', 'POTENTIALLY_CONFUSED'), row.get('LLM防坑口诀', '')
                
                # 混淆属性同理，注入双向边
                self.G.add_edge(src, tgt, relation=rel, rule_desc=rule)
                self.G.add_edge(tgt, src, relation=rel, rule_desc=rule)
            print(f"✅ 载入易混淆口诀边: {len(confuse_df)} 条")

        print(f"🎉 图谱构建完成！总节点数: {self.G.number_of_nodes()}, 总边数: {self.G.number_of_edges()}")

    def save_to_disk(self, filepath: str = "data/cache/graph_rag.pkl"):
        """将图谱序列化保存，下次秒级加载"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.G, f)
        print(f"💾 图谱已缓存至 {filepath}")

    def load_from_disk(self, filepath: str = "data/cache/graph_rag.pkl"):
        """读取缓存的图谱"""
        with open(filepath, 'rb') as f:
            self.G = pickle.load(f)
        print(f"🚀 图谱极速加载完毕！")

    # ================= 供 Retriever (RAG) 调用的四大专属安检 API =================

    def get_job_features(self, code: str) -> Optional[Dict]:
        """【API-1】特征安检：获取职业的原子特征和红线标识"""
        if not self.G.has_node(code):
            return None
        data = self.G.nodes[code]
        if data.get("node_type") != "Job":
            return None
            
        return {
            # ── 旧字段，向后兼容 ──
            "名称": data.get("name", ""),
            "动作": data.get("core_actions", []) or data.get("actions", []),
            "对象": data.get("objects", []),
            "环境": data.get("environment", []) or data.get("envs", []),
            "是否涉公权": data.get("is_gov", False),
            "是否涉临床": data.get("is_med", False),
            # ── 新增 7D 字段 ──
            "deliverables": data.get("deliverables", []),
            "main_kpi": data.get("main_kpi", ""),
            "core_actions": data.get("core_actions", []),
            "environment": data.get("environment", []),
            "served_population": data.get("served_population", ""),
            "role_level": data.get("role_level", ""),
            "category": data.get("category", ""),
        }

    def get_confused_codes(self, code: str) -> List[str]:
        """【API-2】对决雷达：查出与目标代码存在“易混淆关系”的所有代码 (用于强制拉入候选池)"""
        if not self.G.has_node(code):
            return []
        
        confused_list = []
        # 得益于建图时的双向边注入，我们只需查询单向 successor 即可捕获所有对手
        for neighbor in self.G.successors(code):
            if self.G.edges[code, neighbor].get("relation") == "POTENTIALLY_CONFUSED":
                confused_list.append(neighbor)
        return confused_list

    def get_confusion_text(self, code_a: str, code_b: str) -> Optional[str]:
        """【API-3】提取口诀：提取两个代码之间的 LLM 防坑口诀"""
        if self.G.has_edge(code_a, code_b):
            edge_data = self.G.edges[code_a, code_b]
            if edge_data.get("relation") == "POTENTIALLY_CONFUSED":
                return edge_data.get("rule_desc")
        return None

    def check_wormhole(self, code_a: str, code_b: str) -> Optional[str]:
        """【API-4】红线对撞：检查两个代码之间是否存在绝对互斥红线"""
        # 同样得益于双向边注入，原来繁琐的 elif 反查被缩减为极简的单行校验
        if self.G.has_edge(code_a, code_b):
            edge_data = self.G.edges[code_a, code_b]
            if edge_data.get("relation") == "MUTUALLY_EXCLUSIVE":
                return edge_data.get("rule_desc")
        return None