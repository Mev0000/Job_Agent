"""
core/graph_manager.py

GraphRAG 2.0 敏捷图谱管理器
- 提供图谱节点的CRUD接口
- 支持JSON热重载（替代PKL黑盒）
- 注意：需要人工确认的部分只定义接口，不实现具体逻辑
"""

import json
import os
import networkx as nx
from typing import Dict, List, Any, Optional, Set


class AgileGraphManager:
    """
    敏捷图谱管理器
    
    使用说明：
    - 读取接口（get_*）可立即使用
    - 写入接口（update_*, remove_*）仅定义接口，具体实现需人工确认后启用
    - JSON存储格式便于人工编辑和Git版本控制
    """
    
    def __init__(self, cache_dir: str = "data/cache/"):
        """
        初始化图谱管理器
        
        TODO: 需要人工确认后实现
        - cache_dir 路径是否正确
        - 是否需要支持多图谱实例（如按行业分图谱）
        """
        self.cache_dir = cache_dir
        self.G = nx.DiGraph()
        self._load_or_build()
    
    # ================= 读取接口（可立即使用）====================
    
    def get_node_7_dimensions(self, code: str) -> Dict[str, Any]:
        """
        获取节点的七维属性（对齐方案A）
        
        返回格式：
        {
            "node_id": "4-01-02",
            "name": "营销员",
            "role_level": "4类-一线执行",
            "core_actions": ["推销", "客户开发"],
            "objects": ["客户", "产品"],
            "deliverables": ["销售订单", "客户档案"],
            "main_kpi": "成交额",
            "environment": ["线下门店", "市场外场"],
            "served_population": ["消费者", "企业客户"]
        }
        
        TODO: 需要人工确认后实现
        - 当前图谱节点可能缺少部分维度，需要离线蒸馏补全
        - role_level 是否应独立为元数据表（不参与向量相似度计算）
        """
        if self.G.has_node(code):
            node_data = self.G.nodes[code]
            # 返回七维属性（缺失维度返回空列表/默认值）
            return {
                "node_id": code,
                "name": node_data.get("name", ""),
                "role_level": node_data.get("role_level", ""),
                "core_actions": node_data.get("core_actions", []),
                "objects": node_data.get("objects", []),
                "deliverables": node_data.get("deliverables", []),
                "main_kpi": node_data.get("main_kpi", ""),
                "environment": node_data.get("environment", []),
                "served_population": node_data.get("served_population", [])
            }
        return {}
    
    def get_confused_rules(self, code: str) -> List[str]:
        """
        获取节点的易混淆口诀（用于step_4裁决）
        
        TODO: 需要人工确认后实现
        - 口诀文本格式是否需要结构化（如分为"差异点1/差异点2"）
        - 若无口诀，是否触发自动生成（需人工审核）
        """
        rules = []
        if self.G.has_node(code):
            # 遍历所有以code为源的边，提取CONFUSED类型边的rule属性
            for _, target, data in self.G.out_edges(code, data=True):
                if data.get("type") == "CONFUSED":
                    rule_text = data.get("rule", "")
                    if rule_text:
                        rules.append(rule_text)
        return rules
    
    def get_node_features(self, code: str) -> Dict[str, Any]:
        """
        获取节点的基础特征（兼容旧版retriever.py调用）
        
        返回格式（旧版兼容）：
        {
            "动作": [...],
            "对象": [...],
            "环境": [...],
            "是否涉公权": bool,
            "是否涉临床": bool
        }
        """
        if self.G.has_node(code):
            node_data = self.G.nodes[code]
            return {
                "动作": node_data.get("core_actions", []),
                "对象": node_data.get("objects", []),
                "环境": node_data.get("environment", []),
                "是否涉公权": node_data.get("involves_public_power", False),
                "是否涉临床": node_data.get("involves_clinical", False)
            }
        return {}
    
    def check_wormhole(self, code_a: str, code_b: str) -> Optional[str]:
        """
        检查两个节点是否存在虫洞互斥关系
        
        返回：若存在互斥，返回规则文本；否则返回None
        
        TODO: 需要人工确认后实现
        - 虫洞边是否已在图谱中构建（当前可能缺失）
        - 互斥规则是否需要支持"单向互斥"（A禁B但B不禁A）
        """
        if self.G.has_edge(code_a, code_b):
            data = self.G[code_a][code_b]
            if data.get("type") == "WORMHOLE":
                return data.get("rule", "")
        return None
    
    # ================= 写入接口（只定义，需人工确认后实现）====================
    
    def update_expert_rule(self, source_code: str, target_code: str, new_rule: str) -> bool:
        """
        【热更新】更新或创建易混淆口诀边
        
        使用场景：社会学专家在UI界面修改口诀后，后台调用此接口热更新
        
        TODO: 需要人工确认后实现
        - 是否需要版本控制（保留历史口诀）？
        - 更新后是否立即触发图谱保存（还是批量更新后手动保存）？
        - 是否需要对new_rule做合法性校验（如禁止空字符串）？
        
        返回：是否更新成功
        """
        # Interface only - implementation needs human confirmation
        raise NotImplementedError("update_expert_rule() 需要人工确认后实现。请确认：1) 版本控制策略 2) 保存触发机制 3) 合法性校验规则")
    
    def enrich_node_attributes(self, code: str, attr_dict: Dict[str, Any]) -> bool:
        """
        【热更新】增量刷入节点的七维特征
        
        使用场景：离线LLM蒸馏完成后，批量更新节点属性
        
        TODO: 需要人工确认后实现
        - attr_dict的键值是否需要严格校验（防止污染图谱）？
        - 增量更新还是全量覆盖？
        - 更新后是否需要触发图谱质量校验（如七维完整性检查）？
        
        返回：是否更新成功
        """
        # Interface only - implementation needs human confirmation
        raise NotImplementedError("enrich_node_attributes() 需要人工确认后实现。请确认：1) 键值校验规则 2) 更新策略 3) 质量校验机制")
    
    def remove_noisy_edge(self, source: str, target: str, audit_log: bool = True) -> bool:
        """
        【热排雷】删除错误边
        
        使用场景：运行中发现图谱有一条错误连接，瞬间拔除
        
        TODO: 需要人工确认后实现
        - audit_log=True时，审计日志写入哪个文件/数据库？
        - 删除边是否需要备份（防止误删）？
        - 是否需要对删除操作加权限控制（如仅专家可删）？
        
        返回：是否删除成功
        """
        # Interface only - implementation needs human confirmation
        raise NotImplementedError("remove_noisy_edge() 需要人工确认后实现。请确认：1) 审计日志路径 2) 备份机制 3) 权限控制策略")
    
    def add_node_7_dimensions(self, code: str, dimensions: Dict[str, Any]) -> bool:
        """
        【创建节点】添加新的职业节点（含七维属性）
        
        TODO: 需要人工确认后实现
        - 新节点的code是否需要人工审核（防止错误代码入库）？
        - 是否自动构建与其他节点的易混淆边（需人工确认相似度阈值）？
        """
        # Interface only - implementation needs human confirmation
        raise NotImplementedError("add_node_7_dimensions() 需要人工确认后实现。请确认：1) 新节点审核流程 2) 自动建边策略")
    
    # ================= 存储接口（只定义，需人工确认后实现）====================
    
    def save_to_json(self, async_mode: bool = False) -> bool:
        """
        将图谱保存为JSON格式（替代PKL）
        
        TODO: 需要人工确认后实现
        - async_mode=True时，使用threading.Thread包装还是asyncio？
        - JSON Schema校验规则是什么（防止人工编辑引入语法错误）？
        - 保存失败时是否需要回滚机制？
        """
        # Interface only - implementation needs human confirmation
        raise NotImplementedError("save_to_json() 需要人工确认后实现。请确认：1) 异步实现方式 2) Schema校验规则 3) 失败回滚机制")
    
    def load_from_json(self, nodes_file: str = "graph_nodes.json", 
                      edges_file: str = "graph_edges.json") -> bool:
        """
        从JSON文件加载图谱（热重载）
        
        TODO: 需要人工确认后实现
        - 热重载触发机制（mtime检测 vs 手动触发）？
        - JSON文件不存在时，是否自动从PKL迁移？
        - 加载失败时是否回退到PKL？
        """
        # Interface only - implementation needs human confirmation
        raise NotImplementedError("load_from_json() 需要人工确认后实现。请确认：1) 热重载触发机制 2) PKL迁移策略 3) 失败回退策略")
    
    # ================= 内部方法（私有，无需人工确认）====================
    
    def _load_or_build(self):
        """
        加载已有图谱或构建新图谱
        
        P1-G 双轨并行策略（决策已确认）：
        - 阶段一（当前）：GraphManager 作为旁路运行，原 retriever.py 继续读取 PKL 文件
        - 阶段二（待实现）：GraphManager 从 JSON 文件加载，与 PKL 对齐率验证达 100% 后切换
        - 阶段三（上线后）：原 retriever.py 切换为调用 GraphManager，PKL 文件归档备份
        
        当前实现：空初始化，不加载任何数据（旁路模式，不影响主流程）
        TODO: 实现 JSON 加载逻辑（需先确认 JSON 文件格式和节点字段设计）
        """
        # 当前阶段：GraphManager 为旁路空实例，不干扰现有 PKL 流程
        # 待 JSON 数据就绪后，解开下方注释：
        # json_nodes = os.path.join(self.cache_dir, "graph_nodes.json")
        # json_edges = os.path.join(self.cache_dir, "graph_edges.json")
        # if os.path.exists(json_nodes) and os.path.exists(json_edges):
        #     self.load_from_json(json_nodes, json_edges)
        pass
    
    def _validate_7_dimensions(self, dimensions: Dict[str, Any]) -> bool:
        """
        校验七维属性的合法性
        
        TODO: 需要人工确认校验规则
        - core_actions/objects/deliverables等是否为非空列表？
        - role_level是否在合法枚举内？
        """
        # Interface only - validation rules need human confirmation
        return True


# ================= 工厂函数（供retriever.py调用）====================

def create_graph_manager(config: Dict[str, Any]) -> AgileGraphManager:
    """
    工厂函数：根据配置创建图谱管理器
    
    TODO: 需要人工确认后实现
    - config中图谱配置项的key名称
    - 是否需要支持多图谱实例（如按行业分图谱）
    """
    cache_dir = config.get("data", {}).get("graph_cache_dir", "data/cache/")
    return AgileGraphManager(cache_dir=cache_dir)
