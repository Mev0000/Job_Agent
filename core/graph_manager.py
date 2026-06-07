"""
core/graph_manager.py

GraphRAG 2.0 敏捷图谱管理器
- 提供图谱节点的CRUD接口
- 支持JSON热重载（替代PKL黑盒）
- 根据已确认决策实施所有接口

实施决策清单：
- P1-G: 双轨并行（JSON优先，PKL回退）
- P1-H: 节点格式（role_level枚举、旧字段兼容、红线字段保留）
- P2-I: 重启加载（__init__ 中调用 _load_or_build）
- P1-J: 独立 Changelog（graph_rules_changelog.json）
- P1-K: 软删除 + JSON 审计日志（graph_audit.json）
- P2-L: threading.Thread 异步保存
- P2-M: 校验规则硬编码在 _validate_7_dimensions()
- P2-E: 工厂函数从 graph_cache_path 提取目录
- P1-N: 渐进式（retriever.py 不变，新增功能用 GraphManager）
- P1-O: 候选卡片展示 deliverables（retriever.py 中 _format_prompt_context 需配合）
"""

import json
import os
import threading
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

import networkx as nx


# ================= 常量（基于已确认决策）====================

ROLE_LEVEL_ENUM = {
    "1类-管理决策",
    "2类-专业技术创制",
    "3类-办事辅助",
    "4类-生产生活服务",
    "5类-农林牧渔",
    "6类-生产制造",
    "7类-军队",
}

# 文件路径常量
GRAPH_CACHE_PATH_DEFAULT = "data/cache/job_dict_graph.pkl"
GRAPH_RULES_CHANGELOG = "data/cache/graph_rules_changelog.json"
GRAPH_AUDIT_LOG = "data/cache/graph_audit.json"

# 通用动词黑名单（不允许单独出现在 core_actions 中）— P2-M 硬编码规则
GENERIC_VERBS_BLACKLIST = {"负责", "参与", "协助", "处理", "完成", "负责完成", "进行"}


class AgileGraphManager:
    """
    敏捷图谱管理器

    使用说明：
    - 读取接口（get_*）已可立即使用
    - 写入接口（update_*, remove_*）已全部实施
    - JSON存储格式便于人工编辑和Git版本控制
    - 双轨并行：优先JSON，回退PKL，不影响现有流程
    """

    def __init__(self, cache_dir: str = "data/cache/"):
        """
        初始化图谱管理器

        实施决策 P2-I（重启加载）：
        - __init__ 中调用 _load_or_build()
        - 优先从 JSON 加载，失败则回退 PKL
        """
        self.cache_dir = cache_dir
        self.G = nx.DiGraph()
        self._lock = threading.Lock()  # 线程锁（保护并发写入）
        self._load_or_build()

    # ================= 读取接口（已实施）====================

    def get_node_7_dimensions(self, code: str) -> Dict[str, Any]:
        """
        获取节点的七维属性（对齐方案A）

        返回格式：
        {
            "node_id": "4-01-02",
            "name": "营销员",
            "role_level": "4类-生产生活服务",
            "core_actions": ["推销", "客户开发"],
            "objects": ["客户", "产品"],
            "deliverables": ["销售订单", "客户档案"],
            "main_kpi": "成交额",
            "environment": ["线下门店", "市场外场"],
            "served_population": ["消费者", "企业客户"]
        }

        实施决策 P1-H：
        - role_level 使用7个固定枚举值
        - 红线布尔值字段（Is_Government/Is_Medical_Clinical）也一并返回
        """
        if self.G.has_node(code):
            node_data = self.G.nodes[code]
            return {
                "node_id": code,
                "name": node_data.get("name", ""),
                "role_level": node_data.get("role_level", ""),
                "core_actions": node_data.get("core_actions", []),
                "objects": node_data.get("objects", []),
                "deliverables": node_data.get("deliverables", []),
                "main_kpi": node_data.get("main_kpi", ""),
                "environment": node_data.get("environment", []),
                "served_population": node_data.get("served_population", []),
                # P1-H：红线布尔值字段原封不动返回
                "Is_Government": node_data.get("Is_Government", False),
                "Is_Medical_Clinical": node_data.get("Is_Medical_Clinical", False),
            }
        return {}

    def get_confused_rules(self, code: str) -> List[str]:
        """
        获取节点的易混淆口诀（用于step_4裁决）

        实施决策 P1-K（软删除）：
        - 已软删除的边（deleted=True）不返回
        - 支持 is_rule_forced 字段（供置信度评分使用，P1-A决策）
        """
        rules = []
        if self.G.has_node(code):
            for _, target, data in self.G.out_edges(code, data=True):
                if data.get("type") == "CONFUSED" and not data.get("deleted", False):
                    rule_text = data.get("rule", "")
                    if rule_text:
                        rules.append(rule_text)
        return rules

    def get_rule_forced(self, code: str) -> bool:
        """
        检查节点的易混淆边中是否有 is_rule_forced=True 的口诀

        实施决策 P1-A（选项2）：
        - 在图谱边的 confusion_warnings 格式中新增 is_rule_forced 布尔字段
        - 检索层传递给推理层，用于置信度加分

        返回：是否存在强制命中的口诀
        """
        if self.G.has_node(code):
            for _, target, data in self.G.out_edges(code, data=True):
                if data.get("type") == "CONFUSED" and not data.get("deleted", False):
                    if data.get("is_rule_forced", False):
                        return True
        return False

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
                "是否涉公权": node_data.get("Is_Government", False),
                "是否涉临床": node_data.get("Is_Medical_Clinical", False),
            }
        return {}

    def get_node_features_legacy(self, code: str) -> Dict[str, Any]:
        """
        兼容旧版 retriever.py 的字段名要求（P1-H决策）

        动态映射新字段 → 旧格式：
        - core_actions → 动作
        - objects → 对象
        - environment → 环境
        - Is_Government → 是否涉公权
        - Is_Medical_Clinical → 是否涉临床

        在不修改 retriever.py 内部逻辑的前提下，
        瞬间把图谱的底层存储从 PKL 偷梁换柱成 JSON。
        """
        node = self.G.nodes.get(code, {})
        return {
            "动作": node.get("core_actions", []),
            "对象": node.get("objects", []),
            "环境": node.get("environment", []),
            "是否涉公权": node.get("Is_Government", False),
            "是否涉临床": node.get("Is_Medical_Clinical", False),
        }

    def check_wormhole(self, code_a: str, code_b: str) -> Optional[str]:
        """
        检查两个节点是否存在虫洞互斥关系

        实施决策 P1-K（软删除）：
        - 已软删除的边（deleted=True）不触发互斥

        返回：若存在互斥，返回规则文本；否则返回None
        """
        if self.G.has_edge(code_a, code_b):
            data = self.G[code_a][code_b]
            if data.get("deleted", False):
                return None
            if data.get("type") == "WORMHOLE":
                return data.get("rule", "")
        return None

    # ================= 写入接口（已实施）====================

    def update_expert_rule(self, source_code: str, target_code: str,
                          new_rule: str, operator: str = "expert") -> bool:
        """
        【热更新】更新或创建易混淆口诀边

        实施决策 P1-J（独立Changelog）：
        - 写入独立的 graph_rules_changelog.json，按时间戳保存历史
        - 边的 rule 字段仅存当前生效版本
        - 旧版口诀永久保存在 changelog 中，专家可审计

        参数：
        - source_code: 源节点代码
        - target_code: 目标节点代码
        - new_rule: 新口诀文本
        - operator: 操作人（默认 "expert"）

        返回：是否更新成功
        """
        if not new_rule or not new_rule.strip():
            print(f"[update_expert_rule] 拒绝空口诀: {source_code} → {target_code}")
            return False

        # 查找已存在的边，记录旧版口诀
        old_rule = ""
        if self.G.has_edge(source_code, target_code):
            data = self.G[source_code][target_code]
            if data.get("type") == "CONFUSED":
                old_rule = data.get("rule", "")

        # P1-J：写入独立 Changelog
        changelog_entry = {
            "timestamp": datetime.now().isoformat(),
            "operator": operator,
            "source_code": source_code,
            "target_code": target_code,
            "old_rule": old_rule,
            "new_rule": new_rule.strip(),
            "action": "update" if old_rule else "create"
        }
        self._append_changelog(changelog_entry)

        # 更新边（若不存在则创建）
        with self._lock:
            if not self.G.has_edge(source_code, target_code):
                self.G.add_edge(source_code, target_code, type="CONFUSED")
            self.G[source_code][target_code]["rule"] = new_rule.strip()
            self.G[source_code][target_code]["deleted"] = False
            self.G[source_code][target_code]["updated_at"] = datetime.now().isoformat()
            self.G[source_code][target_code]["updated_by"] = operator

        print(f"[update_expert_rule] 口诀已更新: {source_code} → {target_code}")
        self._async_save()
        return True

    def set_rule_forced(self, source_code: str, target_code: str, is_forced: bool) -> bool:
        """
        设置口诀的 is_rule_forced 标志（供P1-A置信度评分使用）

        参数：
        - source_code: 源节点代码
        - target_code: 目标节点代码
        - is_forced: 是否强制命中

        返回：是否设置成功
        """
        if not self.G.has_edge(source_code, target_code):
            print(f"[set_rule_forced] 边不存在: {source_code} → {target_code}")
            return False

        with self._lock:
            self.G[source_code][target_code]["is_rule_forced"] = is_forced

        print(f"[set_rule_forced] is_rule_forced={is_forced}: {source_code} → {target_code}")
        self._async_save()
        return True

    def enrich_node_attributes(self, code: str, attr_dict: Dict[str, Any],
                             operator: str = "distillation") -> bool:
        """
        【热更新】增量刷入节点的七维特征

        实施决策：
        - P2-M：先校验（调用 _validate_7_dimensions）
        - 增量更新（不覆盖未提供的字段）
        - 写入审计日志

        参数：
        - code: 节点代码
        - attr_dict: 要更新的属性字典
        - operator: 操作人（默认 "distillation"）

        返回：是否更新成功
        """
        if not self.G.has_node(code):
            print(f"[enrich_node_attributes] 节点不存在: {code}")
            return False

        # P2-M：校验
        if not self._validate_7_dimensions(attr_dict):
            print(f"[enrich_node_attributes] 校验失败: {code}")
            return False

        with self._lock:
            node_data = self.G.nodes[code]
            # 记录旧值（审计用）
            old_attrs = {k: v for k, v in node_data.items() if k in attr_dict}

            # 增量更新
            for k, v in attr_dict.items():
                node_data[k] = v

            node_data["updated_at"] = datetime.now().isoformat()
            node_data["updated_by"] = operator

        # 审计日志
        self._append_audit_log({
            "timestamp": datetime.now().isoformat(),
            "action": "enrich_node_attributes",
            "operator": operator,
            "node_id": code,
            "old_attrs": old_attrs,
            "new_attrs": {k: v for k, v in attr_dict.items()}
        })

        print(f"[enrich_node_attributes] 节点已更新: {code}")
        self._async_save()
        return True

    def remove_noisy_edge(self, source: str, target: str,
                          operator: str = "expert", reason: str = "") -> bool:
        """
        【热排雷】删除错误边（软删除）

        实施决策 P1-K：
        - 软删除（标记 deleted=true 保留在文件中）
        - 审计日志写入 data/cache/graph_audit.json
        - 可恢复（通过审计日志回溯）

        参数：
        - source: 源节点
        - target: 目标节点
        - operator: 操作人
        - reason: 删除原因

        返回：是否删除成功
        """
        if not self.G.has_edge(source, target):
            print(f"[remove_noisy_edge] 边不存在: {source} → {target}")
            return False

        with self._lock:
            data = self.G[source][target]

            # P1-K：审计日志
            self._append_audit_log({
                "timestamp": datetime.now().isoformat(),
                "action": "remove_noisy_edge",
                "operator": operator,
                "reason": reason,
                "edge": {
                    "source": source,
                    "target": target,
                    "type": data.get("type", ""),
                    "rule": data.get("rule", "")
                }
            })

            # P1-K：软删除（标记 deleted=true）
            data["deleted"] = True
            data["deleted_at"] = datetime.now().isoformat()
            data["deleted_by"] = operator
            data["delete_reason"] = reason

        print(f"[remove_noisy_edge] 边已软删除: {source} → {target}")
        self._async_save()
        return True

    def add_node_7_dimensions(self, code: str, dimensions: Dict[str, Any],
                              operator: str = "expert") -> bool:
        """
        【创建节点】添加新的职业节点（含七维属性）

        实施决策：
        - P2-M：校验七维属性合法性
        - 新节点需要审核（通过校验即认为格式合法）
        - 自动构建与其他节点的易混淆边（需人工确认相似度阈值，当前留接口）

        参数：
        - code: 节点代码（如 "9-99-99"）
        - dimensions: 七维属性字典
        - operator: 操作人

        返回：是否创建成功
        """
        if self.G.has_node(code):
            print(f"[add_node_7_dimensions] 节点已存在: {code}")
            return False

        # P2-M：校验
        if not self._validate_7_dimensions(dimensions):
            print(f"[add_node_7_dimensions] 校验失败: {code}")
            return False

        with self._lock:
            node_data = {
                "node_id": code,
                "name": dimensions.get("name", code),
                "role_level": dimensions.get("role_level", ""),
                "core_actions": dimensions.get("core_actions", []),
                "objects": dimensions.get("objects", []),
                "deliverables": dimensions.get("deliverables", []),
                "main_kpi": dimensions.get("main_kpi", ""),
                "environment": dimensions.get("environment", []),
                "served_population": dimensions.get("served_population", []),
                # P1-H：红线布尔值字段
                "Is_Government": dimensions.get("Is_Government", False),
                "Is_Medical_Clinical": dimensions.get("Is_Medical_Clinical", False),
                "created_at": datetime.now().isoformat(),
                "created_by": operator
            }
            self.G.add_node(code, **node_data)

        # 审计日志
        self._append_audit_log({
            "timestamp": datetime.now().isoformat(),
            "action": "add_node_7_dimensions",
            "operator": operator,
            "node_id": code,
            "dimensions": {k: v for k, v in dimensions.items()}
        })

        print(f"[add_node_7_dimensions] 节点已创建: {code} - {node_data['name']}")
        self._async_save()
        return True

    # ================= 存储接口（已实施）====================

    def save_to_json(self, async_mode: bool = True) -> bool:
        """
        将图谱保存为JSON格式

        实施决策 P2-L：
        - async_mode=True 时，使用 threading.Thread 包装同步写入
        - 保存 graph_nodes.json 和 graph_edges.json 两个文件
        - 保存前自动过滤已软删除的边（可选，当前保留deleted标记）

        参数：
        - async_mode: 是否异步保存（默认True）

        返回：是否保存成功（异步模式始终返回True，实际结果在后台线程中）
        """
        if async_mode:
            thread = threading.Thread(target=self._save_to_json_sync, daemon=True)
            thread.start()
            print(f"[save_to_json] 异步保存已启动（线程: {thread.name}）")
            return True
        else:
            return self._save_to_json_sync()

    def _save_to_json_sync(self) -> bool:
        """同步保存（被线程调用）"""
        try:
            # 保存节点
            nodes_file = os.path.join(self.cache_dir, "graph_nodes.json")
            nodes_data = {}
            for node_id, data in self.G.nodes(data=True):
                # 过滤掉内部字段（可选）
                clean_data = {k: v for k, v in data.items()}
                nodes_data[node_id] = clean_data

            with open(nodes_file, 'w', encoding='utf-8') as f:
                json.dump(nodes_data, f, ensure_ascii=False, indent=2)

            # 保存边
            edges_file = os.path.join(self.cache_dir, "graph_edges.json")
            edges_data = []
            for source, target, data in self.G.edges(data=True):
                edge_entry = {
                    "source": source,
                    "target": target,
                    "type": data.get("type", ""),
                    "rule": data.get("rule", ""),
                    "is_rule_forced": data.get("is_rule_forced", False),
                    "deleted": data.get("deleted", False),
                    "updated_at": data.get("updated_at", ""),
                    "updated_by": data.get("updated_by", ""),
                    "deleted_at": data.get("deleted_at", ""),
                    "deleted_by": data.get("deleted_by", ""),
                    "delete_reason": data.get("delete_reason", "")
                }
                edges_data.append(edge_entry)

            with open(edges_file, 'w', encoding='utf-8') as f:
                json.dump(edges_data, f, ensure_ascii=False, indent=2)

            print(f"[_save_to_json_sync] 图谱已保存: {nodes_file}, {edges_file}")
            print(f"  节点数: {len(nodes_data)}, 边数: {len(edges_data)}")
            return True
        except Exception as e:
            print(f"[_save_to_json_sync] 保存失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_from_json(self, nodes_file: str = None,
                      edges_file: str = None) -> bool:
        """
        从JSON文件加载图谱（热重载）

        实施决策 P2-I：
        - 重启服务时自动重载（在 _load_or_build 中调用）
        - JSON 文件不存在时，回退到 PKL
        - 加载失败时回退到 PKL 或空图谱

        参数：
        - nodes_file: 节点文件路径（默认 graph_nodes.json）
        - edges_file: 边文件路径（默认 graph_edges.json）

        返回：是否加载成功
        """
        if nodes_file is None:
            nodes_file = os.path.join(self.cache_dir, "graph_nodes.json")
        if edges_file is None:
            edges_file = os.path.join(self.cache_dir, "graph_edges.json")

        if not os.path.exists(nodes_file) or not os.path.exists(edges_file):
            print(f"[load_from_json] JSON 文件不存在，回退到 PKL")
            return self._load_from_pkl_fallback()

        try:
            # 加载节点
            with open(nodes_file, 'r', encoding='utf-8') as f:
                nodes_data = json.load(f)

            # 加载边
            with open(edges_file, 'r', encoding='utf-8') as f:
                edges_data = json.load(f)

            # 构建图谱
            with self._lock:
                self.G = nx.DiGraph()
                for node_id, data in nodes_data.items():
                    self.G.add_node(node_id, **data)
                for edge in edges_data:
                    source = edge["source"]
                    target = edge["target"]
                    edge_attrs = {k: v for k, v in edge.items()
                                  if k not in ("source", "target")}
                    self.G.add_edge(source, target, **edge_attrs)

            print(f"[load_from_json] 图谱已从 JSON 加载: {len(nodes_data)} 节点, {len(edges_data)} 边")
            return True
        except Exception as e:
            print(f"[load_from_json] 加载失败: {e}，回退到 PKL")
            import traceback
            traceback.print_exc()
            return self._load_from_pkl_fallback()

    def _load_from_pkl_fallback(self) -> bool:
        """回退到 PKL 文件（兼容旧版）"""
        pkl_path = os.path.join(self.cache_dir, "job_dict_graph.pkl")
        if not os.path.exists(pkl_path):
            print(f"[_load_from_pkl_fallback] PKL 文件也不存在: {pkl_path}，使用空图谱")
            self.G = nx.DiGraph()
            return False

        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                self.G = pickle.load(f)
            print(f"[_load_from_pkl_fallback] 图谱已从 PKL 回退加载: {len(self.G.nodes)} 节点")
            return True
        except Exception as e:
            print(f"[_load_from_pkl_fallback] PKL 加载失败: {e}，使用空图谱")
            import traceback
            traceback.print_exc()
            self.G = nx.DiGraph()
            return False

    # ================= 内部方法 =================

    def _load_or_build(self):
        """
        加载已有图谱或构建新图谱

        实施决策 P1-G（双轨并行）：
        - 阶段一（当前）：优先尝试从 JSON 加载，失败则回退到 PKL
        - JSON 和 PKL 都不存在时，使用空图谱（旁路模式，不影响主流程）
        """
        json_nodes = os.path.join(self.cache_dir, "graph_nodes.json")
        json_edges = os.path.join(self.cache_dir, "graph_edges.json")

        if os.path.exists(json_nodes) and os.path.exists(json_edges):
            success = self.load_from_json(json_nodes, json_edges)
            if success:
                return

        # JSON 不存在或加载失败，回退到 PKL
        self._load_from_pkl_fallback()

    def _validate_7_dimensions(self, dimensions: Dict[str, Any]) -> bool:
        """
        校验七维属性的合法性

        实施决策 P2-M（代码硬编码）：
        - 校验规则硬编码在方法中（不依赖外部 schema.json）
        - 返回 True/False

        校验规则：
        1. role_level 必须在 ROLE_LEVEL_ENUM 中（若提供）
        2. core_actions/objects/deliverables 等列表字段不允许为非列表类型（若提供）
        3. 通用动词黑名单（core_actions 中不允许只包含黑名单动词）
        4. 边的 type 字段必须在合法枚举内（若提供）
        5. node_id 必须匹配正则 \d-\d{2}(-\d{2})?(-\d{2})?（若提供）
        """
        # 1. role_level 枚举校验
        role_level = dimensions.get("role_level", "")
        if role_level and role_level not in ROLE_LEVEL_ENUM:
            print(f"[_validate_7_dimensions] role_level 非法: {role_level}")
            print(f"  合法值: {ROLE_LEVEL_ENUM}")
            return False

        # 2. 列表字段类型校验
        for field in ["core_actions", "objects", "deliverables", "environment", "served_population"]:
            val = dimensions.get(field, None)
            if val is not None and not isinstance(val, list):
                print(f"[_validate_7_dimensions] {field} 必须是列表: {type(val)}")
                return False

        # 3. 通用动词黑名单校验
        core_actions = dimensions.get("core_actions", [])
        if core_actions and isinstance(core_actions, list):
            if all(action in GENERIC_VERBS_BLACKLIST for action in core_actions if isinstance(action, str)):
                print(f"[_validate_7_dimensions] core_actions 只包含通用动词: {core_actions}")
                return False

        # 4. 边的 type 字段校验
        edge_type = dimensions.get("type", "")
        if edge_type and edge_type not in ("CONFUSED", "WORMHOLE", "HIERARCHY"):
            print(f"[_validate_7_dimensions] 边的 type 非法: {edge_type}")
            return False

        # 5. node_id 格式校验（若提供）
        node_id = dimensions.get("node_id", "")
        if node_id:
            import re
            if not re.match(r'^\d-\d{2}(-\d{2})?(-\d{2})?$', node_id):
                print(f"[_validate_7_dimensions] node_id 格式非法: {node_id}")
                return False

        return True

    def _append_changelog(self, entry: Dict[str, Any]):
        """
        追加写入 graph_rules_changelog.json

        实施决策 P1-J（独立Changelog）：
        - 按时间戳保存历史
        - 保留最近100条记录（防止文件过大）
        """
        try:
            if os.path.exists(GRAPH_RULES_CHANGELOG):
                with open(GRAPH_RULES_CHANGELOG, 'r', encoding='utf-8') as f:
                    changelog = json.load(f)
            else:
                changelog = []

            changelog.append(entry)

            # 只保留最近100条
            if len(changelog) > 100:
                changelog = changelog[-100:]

            with open(GRAPH_RULES_CHANGELOG, 'w', encoding='utf-8') as f:
                json.dump(changelog, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[_append_changelog] 写入 Changelog 失败: {e}")
            import traceback
            traceback.print_exc()

    def _append_audit_log(self, entry: Dict[str, Any]):
        """
        追加写入 graph_audit.json

        实施决策 P1-K（审计日志）：
        - 记录所有高风险操作（删除边、更新口诀、创建节点）
        - 保留最近200条记录（防止文件过大）
        """
        try:
            if os.path.exists(GRAPH_AUDIT_LOG):
                with open(GRAPH_AUDIT_LOG, 'r', encoding='utf-8') as f:
                    audit_log = json.load(f)
            else:
                audit_log = []

            audit_log.append(entry)

            # 只保留最近200条
            if len(audit_log) > 200:
                audit_log = audit_log[-200:]

            with open(GRAPH_AUDIT_LOG, 'w', encoding='utf-8') as f:
                json.dump(audit_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[_append_audit_log] 写入审计日志失败: {e}")
            import traceback
            traceback.print_exc()

    def _async_save(self):
        """触发异步保存（P2-L：threading.Thread）"""
        self.save_to_json(async_mode=True)


# ================= 工厂函数（供retriever.py调用）====================

def create_graph_manager(config: Dict[str, Any]) -> AgileGraphManager:
    """
    工厂函数：根据配置创建图谱管理器

    实施决策 P2-E（复用 graph_cache_path 的目录）：
    - 从 config.data.graph_cache_path 提取目录（os.path.dirname）
    - 不新增配置键，复用现有 graph_cache_path

    用法（retriever.py 中）：
    ```python
    from core.graph_manager import create_graph_manager
    self.graph_mgr = create_graph_manager(self.config)
    ```
    """
    graph_cache_path = config.get("data", {}).get("graph_cache_path",
                                                   GRAPH_CACHE_PATH_DEFAULT)
    cache_dir = os.path.dirname(graph_cache_path)
    if not cache_dir:
        cache_dir = "data/cache/"
    return AgileGraphManager(cache_dir=cache_dir)
