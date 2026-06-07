#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证重建后的图谱质量

检查维度:
    1. 节点覆盖 — 三级/四级/Entity 节点数量
    2. 7D 字段填充率 — 每个字段的覆盖率
    3. 边完整性 — BELONGS_TO / POTENTIALLY_CONFUSED / MUTUALLY_EXCLUSIVE 数量
    4. 层级链路 — 孤儿节点检测、四级→三级→二级链路
    5. 交叉验证 — 图谱节点 vs 蒸馏源数据一致性

使用方式:
    python scripts/verify_rebuild.py
    python scripts/verify_rebuild.py --output report.json
    python scripts/verify_rebuild.py --pkl data/cache/graph_rag.pkl
"""

import json
import os
import sys
import pickle
import argparse
import random
from collections import defaultdict

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_graph(pkl_path: str):
    """加载 PKL 图谱"""
    if not os.path.exists(pkl_path):
        print(f"❌ 图谱不存在: {pkl_path}")
        sys.exit(1)
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def verify_nodes(G):
    """维度 1 + 2: 节点覆盖 + 7D 字段填充率"""
    print("=" * 60)
    print("📊 维度 1+2: 节点覆盖与字段填充率")
    print("=" * 60)

    stats = {"total": G.number_of_nodes()}

    # 按类型统计
    job_cnt = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "Job")
    entity_cnt = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "Entity")
    other_cnt = stats["total"] - job_cnt - entity_cnt

    # 按层级统计 Job
    l3 = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Job" and d.get("level") == "三级"]
    l4 = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Job" and d.get("level") == "四级"]

    print(f"   总节点: {stats['total']}")
    print(f"   Job 节点: {job_cnt}（三级:{len(l3)}, 四级:{len(l4)}）")
    print(f"   Entity 节点: {entity_cnt}")
    print(f"   其他节点: {other_cnt}")

    # 7D 字段填充率（仅四级节点）
    seven_d_fields = ["core_actions", "objects", "deliverables", "main_kpi",
                      "environment", "served_population", "role_level", "category"]

    field_counts = defaultdict(int)
    for code in l4:
        node = G.nodes[code]
        for f in seven_d_fields:
            val = node.get(f)
            if val is not None and val != [] and val != "":
                field_counts[f] += 1

    print(f"\n   7D 字段填充率（{len(l4)} 个四级节点）:")
    for f in seven_d_fields:
        cnt = field_counts[f]
        pct = cnt / len(l4) * 100 if l4 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"     {f:<20s} {bar} {cnt}/{len(l4)} ({pct:.1f}%)")

    stats["l3_count"] = len(l3)
    stats["l4_count"] = len(l4)
    stats["entity_count"] = entity_cnt
    stats["field_fill"] = dict(field_counts)

    return stats


def verify_edges(G):
    """维度 3: 边完整性"""
    print("\n" + "=" * 60)
    print("🔗 维度 3: 边完整性")
    print("=" * 60)

    edge_types = defaultdict(int)
    for u, v, data in G.edges(data=True):
        rel = data.get("relation", "unknown")
        edge_types[rel] += 1

    total_edges = G.number_of_edges()
    print(f"   总边数: {total_edges}")

    expected = {
        "BELONGS_TO": "层级从属",
        "POTENTIALLY_CONFUSED": "易混淆（双向×2）",
        "MUTUALLY_EXCLUSIVE": "法理红线（双向×2）",
        "has_entity": "实体关联",
    }
    legacy_entity = {
        "OPERATES_IN_ENV": "旧版NER·操作环境",
        "INVOLVES_ACTION": "旧版NER·涉及动作",
        "TARGETS_OBJECT": "旧版NER·对象目标",
    }

    for rel_type, desc in expected.items():
        cnt = edge_types.get(rel_type, 0)
        # 混淆和虫洞是双向的，实际 CSV 行数是 cnt/2
        if rel_type in ("POTENTIALLY_CONFUSED", "MUTUALLY_EXCLUSIVE"):
            print(f"   {rel_type:<25s} {cnt:>8d} 条  ({cnt//2} 对, 双向) — {desc}")
        else:
            print(f"   {rel_type:<25s} {cnt:>8d} 条  — {desc}")

    # 旧版NER实体边（推理管线不查询，保留不动）
    for rel_type, desc in legacy_entity.items():
        cnt = edge_types.get(rel_type, 0)
        if cnt > 0:
            print(f"   {rel_type:<25s} {cnt:>8d} 条  — {desc}（旧版，保留）")

    # 真正的未知类型
    all_known = set(expected.keys()) | set(legacy_entity.keys())
    for rel_type, cnt in edge_types.items():
        if rel_type not in all_known:
            print(f"   {rel_type:<25s} {cnt:>8d} 条  — ⚠️ 未知类型")

    return {"total_edges": total_edges, "by_type": dict(edge_types)}


def verify_hierarchy(G):
    """维度 4: 层级链路"""
    print("\n" + "=" * 60)
    print("🏛️  维度 4: 层级链路")
    print("=" * 60)

    # 找出所有四级节点，检查是否都有 parent (BELONGS_TO 出边)
    l4_codes = [n for n, d in G.nodes(data=True)
                if d.get("node_type") == "Job" and d.get("level") == "四级"]

    orphans = []
    good_chain = 0
    for code in l4_codes:
        parent = None
        for succ in G.successors(code):
            if G.edges[code, succ].get("relation") == "BELONGS_TO":
                parent = succ
                break
        if parent:
            # 四级 → 三级
            grandparent = None
            for succ in G.successors(parent):
                if G.edges[parent, succ].get("relation") == "BELONGS_TO":
                    grandparent = succ
                    break
            if grandparent:
                good_chain += 1
            else:
                orphans.append((code, "四级→三级缺失祖父"))
        else:
            orphans.append((code, "四级→三级缺失父节点"))

    print(f"   四级节点: {len(l4_codes)}")
    print(f"   链路完整（四级→三级→二级）: {good_chain}/{len(l4_codes)}")
    if orphans:
        print(f"   ⚠️ 孤儿节点: {len(orphans)} 个")
        for code, reason in orphans[:10]:
            name = G.nodes[code].get("name", "?")
            print(f"     {code} {name} — {reason}")

    return {"l4_total": len(l4_codes), "chain_intact": good_chain, "orphans": len(orphans)}


def verify_field_quality(G, l4_codes):
    """辅助: 字段质量检查"""
    print("\n" + "=" * 60)
    print("✅ 维度 5: 字段质量")
    print("=" * 60)

    # 字段质量检查（蒸馏prompt自由生成枚举值，不做硬编码校验）
    empty_role_level = 0
    empty_core_actions = 0
    empty_objects = 0
    role_level_dist = defaultdict(int)

    for code in l4_codes:
        node = G.nodes[code]
        rl = node.get("role_level", "")
        role_level_dist[rl] += 1 if rl else 0
        if not rl:
            empty_role_level += 1
        if not node.get("core_actions"):
            empty_core_actions += 1
        if not node.get("objects"):
            empty_objects += 1

    print(f"   role_level 为空: {empty_role_level}")
    print(f"   role_level 分布 (top 10):")
    for rl, cnt in sorted(role_level_dist.items(), key=lambda x: -x[1])[:10]:
        print(f"     {rl:<16s} {cnt} 个")
    print(f"   core_actions 为空: {empty_core_actions}（应为2，来自LOW质量节点）")
    print(f"   objects 为空: {empty_objects}（应为2，来自LOW质量节点）")

    # 采样展示
    if l4_codes:
        samples = random.sample(l4_codes, min(3, len(l4_codes)))
        print(f"\n   采样展示:")
        for code in sorted(samples):
            node = G.nodes[code]
            print(f"   {code} — {node.get('name', '?')}")
            print(f"     core_actions ({len(node.get('core_actions',[]))}): {node.get('core_actions',[])[:3]}")
            print(f"     objects ({len(node.get('objects',[]))}): {node.get('objects',[])[:3]}")
            print(f"     deliverables ({len(node.get('deliverables',[]))}): {node.get('deliverables',[])[:3]}")
            print(f"     main_kpi: {node.get('main_kpi', '')}")
            print(f"     environment ({len(node.get('environment',[]))}): {node.get('environment',[])[:3]}")
            print(f"     served_population ({len(node.get('served_population',[]))}): {node.get('served_population',[])[:3]}")
            print(f"     role_level: {node.get('role_level', '')}")
            print(f"     category: {node.get('category', '')}")

    return {
        "empty_role_level": empty_role_level,
        "empty_core_actions": empty_core_actions,
        "empty_objects": empty_objects,
        "role_level_options": len(role_level_dist),
    }


def verify_cross(G, distilled_path: str):
    """交叉验证: 图谱节点 vs 蒸馏源"""
    print("\n" + "=" * 60)
    print("🔀 交叉验证: 图谱 vs 蒸馏源")
    print("=" * 60)

    if not os.path.exists(distilled_path):
        print(f"   ⏭️  蒸馏源不存在 ({distilled_path})，跳过")
        return {}

    with open(distilled_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        distilled = {r.get("node_id", ""): r for r in raw if r.get("node_id")}
    else:
        distilled = raw

    l4_codes = [n for n, d in G.nodes(data=True)
                if d.get("node_type") == "Job" and d.get("level") == "四级"]

    # 双向差异
    in_graph_not_distilled = set(l4_codes) - set(distilled.keys())
    in_distilled_not_graph = set(distilled.keys()) - set(l4_codes)

    print(f"   四级节点(图谱): {len(l4_codes)}, 蒸馏数据: {len(distilled)}")
    print(f"   交集: {len(set(l4_codes) & set(distilled.keys()))}")
    if in_graph_not_distilled:
        print(f"   ⚠️ 图谱有但蒸馏无: {len(in_graph_not_distilled)} 个")
    if in_distilled_not_graph:
        print(f"   ⚠️ 蒸馏有但图谱无: {len(in_distilled_not_graph)} 个")

    # 字段值一致性抽查
    overlap = list(set(l4_codes) & set(distilled.keys()))
    if overlap:
        samples = random.sample(overlap, min(10, len(overlap)))
        mismatches = 0
        for code in samples:
            node = G.nodes[code]
            d7 = distilled[code]
            if node.get("core_actions") != d7.get("core_actions"):
                mismatches += 1
            if node.get("objects") != d7.get("objects"):
                mismatches += 1
        print(f"   字段一致性抽查 ({len(samples)} 个): {len(samples) - mismatches}/{len(samples)} 完全一致")

    return {
        "l4_graph": len(l4_codes),
        "l4_distilled": len(distilled),
        "intersection": len(set(l4_codes) & set(distilled.keys())),
        "graph_only": len(in_graph_not_distilled),
        "distilled_only": len(in_distilled_not_graph),
    }


def main():
    parser = argparse.ArgumentParser(description="验证重建后的图谱质量")
    parser.add_argument("--pkl", type=str, default="data/cache/graph_rag.pkl",
                        help="图谱 PKL 路径")
    parser.add_argument("--distilled", type=str, default="data/cache/graph_nodes.json",
                        help="蒸馏 7D 数据（用于交叉验证）")
    parser.add_argument("--output", type=str, default=None,
                        help="导出 JSON 报告路径")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    pkl_path = os.path.join(project_root, args.pkl)
    distilled_path = os.path.join(project_root, args.distilled)

    print(f"📂 图谱: {pkl_path}")
    print(f"📂 蒸馏源: {distilled_path}")

    G = load_graph(pkl_path)

    report = {}

    # 维度 1+2
    report["nodes"] = verify_nodes(G)

    # 维度 3
    report["edges"] = verify_edges(G)

    # 维度 4
    report["hierarchy"] = verify_hierarchy(G)

    # 维度 5
    l4_codes = [n for n, d in G.nodes(data=True)
                if d.get("node_type") == "Job" and d.get("level") == "四级"]
    report["field_quality"] = verify_field_quality(G, l4_codes)

    # 交叉验证
    if os.path.exists(distilled_path):
        report["cross_validation"] = verify_cross(G, distilled_path)

    # 总结
    print("\n" + "=" * 60)
    print("📋 总结")
    print("=" * 60)
    n = report["nodes"]
    e = report["edges"]
    h = report["hierarchy"]
    fq = report["field_quality"]

    issues = []
    if n["l4_count"] != 1676:
        issues.append(f"四级节点数异常: {n['l4_count']} (期望 1676)")
    if n["l3_count"] != 450:
        issues.append(f"三级节点数异常: {n['l3_count']} (期望 450)")

    # core_actions预期覆盖1674（2个LOW节点跳过）
    expect_ok = 1674
    actual_ca = n["field_fill"].get("core_actions", 0)
    if actual_ca < expect_ok:
        issues.append(f"core_actions 覆盖不足: {actual_ca}/{n['l4_count']} (期望≥{expect_ok})")

    if h["orphans"] > 0:
        issues.append(f"存在 {h['orphans']} 个层级孤儿节点")
    if fq["empty_role_level"] > 0:
        issues.append(f"存在 {fq['empty_role_level']} 个空 role_level")
    if fq["empty_core_actions"] > 2:
        issues.append(f"空 core_actions 超过预期: {fq['empty_core_actions']} (期望≤2)")
    if fq["empty_objects"] > 2:
        issues.append(f"空 objects 超过预期: {fq['empty_objects']} (期望≤2)")

    if issues:
        print("⚠️ 发现以下问题:")
        for i in issues:
            print(f"   - {i}")
    else:
        print("✅ 全部验证通过！")

    # 导出报告
    if args.output:
        report_path = os.path.join(project_root, args.output)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📄 报告已导出: {report_path}")


if __name__ == "__main__":
    main()
