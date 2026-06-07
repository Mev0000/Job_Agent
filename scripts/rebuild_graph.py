#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重建图谱：合并 CSV 基础数据 + 蒸馏7D特征 → 输出 PKL

使用方式:
    python scripts/rebuild_graph.py                          # 全量重建
    python scripts/rebuild_graph.py --include-low            # 包含 LOW 质量节点
    python scripts/rebuild_graph.py --dry-run                # 仅预览，不保存
    python scripts/rebuild_graph.py --output cache/my.pkl    # 自定义输出路径

输入:
    data/graph_tables/*.csv          — 5 张建图 CSV
    data/cache/graph_nodes.json      — 蒸馏 7D 输出 (1676 条四级节点)

输出:
    data/cache/graph_rag.pkl         — 含完整 7D 字段的 networkx DiGraph
"""

import json
import os
import sys
import argparse

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_builder import DictGraphRAG


def main():
    parser = argparse.ArgumentParser(description="重建大典职业图谱（含7D蒸馏数据）")
    parser.add_argument("--include-low", action="store_true",
                        help="包含 _quality==LOW 的节点（默认跳过）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅预览匹配情况，不保存文件")
    parser.add_argument("--output", type=str, default="data/cache/graph_rag.pkl",
                        help="输出 PKL 路径（默认 data/cache/graph_rag.pkl）")
    parser.add_argument("--data-dir", type=str, default="data/graph_tables/",
                        help="CSV 建图表目录")
    parser.add_argument("--distilled", type=str, default="data/cache/graph_nodes.json",
                        help="蒸馏 7D 数据路径")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    distilled_path = os.path.join(project_root, args.distilled)

    # ── 1. 加载蒸馏数据 ──
    print("=" * 60)
    print("📦 加载蒸馏数据...")
    if not os.path.exists(distilled_path):
        print(f"❌ 蒸馏数据不存在: {distilled_path}")
        print("   请先运行: python scripts/distill_7d.py")
        sys.exit(1)

    with open(distilled_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        distilled = {}
        for item in raw:
            code = item.get("node_id", "")
            if code:
                distilled[code] = item
    else:
        distilled = raw

    total_7d = len(distilled)
    low_count = sum(1 for v in distilled.values() if v.get("_quality") == "LOW")
    ok_count = total_7d - low_count
    print(f"   蒸馏条目: {total_7d}（OK:{ok_count}, LOW:{low_count}）")

    # ── 2. Dry-run 预览 ──
    if args.dry_run:
        print("\n🔍 Dry-run 模式 — 预览节点匹配情况...\n")

        # 先建不带7D的图谱，统计节点
        rag = DictGraphRAG(data_dir=os.path.join(project_root, args.data_dir))
        rag.build_graph(seven_d_data=None)

        job_nodes = [n for n, d in rag.G.nodes(data=True) if d.get("node_type") == "Job"]
        l4_nodes = [n for n, d in rag.G.nodes(data=True)
                    if d.get("node_type") == "Job" and d.get("level") == "四级"]
        l3_nodes = [n for n, d in rag.G.nodes(data=True)
                    if d.get("node_type") == "Job" and d.get("level") == "三级"]

        print(f"   CSV 图谱节点:")
        print(f"      Job 节点总计: {len(job_nodes)}")
        print(f"      三级节点: {len(l3_nodes)}")
        print(f"      四级节点: {len(l4_nodes)}")
        print(f"      Entity 节点: {rag.G.number_of_nodes() - len(job_nodes)}")
        print()

        # 匹配分析
        matched = set(l4_nodes) & set(distilled.keys())
        unmatched_in_graph = set(l4_nodes) - set(distilled.keys())
        unmatched_in_distilled = set(distilled.keys()) - set(l4_nodes)

        print(f"   匹配情况:")
        print(f"      匹配成功: {len(matched)}/{len(l4_nodes)}")
        if unmatched_in_graph:
            print(f"      图谱有但蒸馏无: {len(unmatched_in_graph)}")
            for c in sorted(list(unmatched_in_graph))[:5]:
                print(f"        {c}")
        if unmatched_in_distilled:
            print(f"      蒸馏有但图谱无: {len(unmatched_in_distilled)}")
            for c in sorted(list(unmatched_in_distilled))[:5]:
                print(f"        {c}")

        # 采样展示
        if matched:
            sample_codes = sorted(list(matched))[:3]
            print(f"\n   采样预览 (前 3 个):")
            for code in sample_codes:
                d7 = distilled[code]
                print(f"     {code} — {d7.get('name', '?')}")
                print(f"       core_actions: {len(d7.get('core_actions', []))} 条")
                print(f"       objects: {len(d7.get('objects', []))} 条")
                print(f"       deliverables: {len(d7.get('deliverables', []))} 条")
                print(f"       main_kpi: {repr(d7.get('main_kpi', ''))}")
                print(f"       environment: {len(d7.get('environment', []))} 条")
                print(f"       served_population: {len(d7.get('served_population', []))} 条")
                print(f"       role_level: {repr(d7.get('role_level', ''))}")
                print(f"       category: {repr(d7.get('category', ''))}")
        return

    # ── 3. 处理 LOW 质量 ──
    if not args.include_low and low_count > 0:
        print(f"⏭️  跳过 {low_count} 个 LOW 质量节点（使用 --include-low 可包含）")

    # ── 4. 重建图谱 ──
    print("\n🚀 重建图谱...")
    rag = DictGraphRAG(data_dir=os.path.join(project_root, args.data_dir))
    rag.build_graph(seven_d_data=distilled)

    # ── 5. 保存 ──
    output_path = os.path.join(project_root, args.output)
    rag.save_to_disk(output_path)

    print(f"\n✅ 图谱重建完成 → {output_path}")


if __name__ == "__main__":
    main()
