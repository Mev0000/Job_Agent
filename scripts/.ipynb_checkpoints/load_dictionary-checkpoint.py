"""
scripts/load_dictionary.py
解析《2022年职业分类大典》CSV 文件，输出标准化四级节点清单。

功能：
  1. 读取大典CSV（含一级~四级所有层级）
  2. 筛选四级节点（职业编码含3个短横线）
  3. 提取父类编码 / 二级前缀 / 类别标记
  4. 输出 data/cache/dict_level4.csv

使用：
  python scripts/load_dictionary.py
  python scripts/load_dictionary.py --csv data/raw_jd/2022年职业分类大典（整体修订）.csv
"""

import os
import sys
import argparse
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DICT_CSV = os.path.join(PROJECT_ROOT, "data", "raw_jd", "2022年职业分类大典（整体修订）.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "cache", "dict_level4.csv")


def _get_category(code: str) -> str:
    """从编码前缀判定职业大类"""
    first = code.split("-")[0].strip()
    mapping = {
        "1": "1类-管理决策",
        "2": "2类-专业技术创制",
        "3": "3类-办事辅助",
        "4": "4类-生产生活服务",
        "5": "5类-农林牧渔",
        "6": "6类-生产制造",
        "7": "7类-军队",
    }
    return mapping.get(first, "8类-不便分类")


def _get_parent_code(code: str) -> str:
    """获取父类编码（四级→三级），如 1-01-00-01 → 1-01-00"""
    parts = code.split("-")
    if len(parts) == 4:
        return "-".join(parts[:3])
    return ""


def _get_l2_prefix(code: str) -> str:
    """获取二级前缀，如 1-01-00-01 → 1-01"""
    parts = code.split("-")
    return "-".join(parts[:2])


def load_and_filter(dict_csv: str, output_csv: str):
    """加载大典CSV，筛选四级节点，输出标准格式"""
    if not os.path.exists(dict_csv):
        print(f"❌ 找不到大典CSV：{dict_csv}")
        print(f"   请确认文件路径，或将Excel导出为CSV后重新运行")
        return None

    print(f"📖 读取大典CSV：{dict_csv}")
    df = pd.read_csv(dict_csv).fillna("")

    # 确保列名正确
    expected_cols = ["层级", "职业编码", "职业名称", "职业描述", "主要工作任务"]
    for col in expected_cols:
        if col not in df.columns:
            print(f"❌ 缺少列：{col}，实际列：{list(df.columns)}")
            return None

    # 筛选四级节点（编码含3个短横线，即4位编码）
    df["dash_count"] = df["职业编码"].apply(lambda c: str(c).count("-"))
    df_l4 = df[df["dash_count"] == 3].copy()
    df_l4.drop(columns=["dash_count"], inplace=True)

    print(f"✅ 全量大典：{len(df)} 行 → 四级节点：{len(df_l4)} 行")
    print(f"   层级分布：{df['层级'].value_counts().to_dict()}")

    # 添加衍生字段（保留大典原始列名以兼容 distill_7d.py）
    df_l4["code"] = df_l4["职业编码"].astype(str).str.strip()
    df_l4["name"] = df_l4["职业名称"].astype(str).str.strip()
    df_l4["desc"] = df_l4["职业描述"].astype(str).str.strip()
    df_l4["tasks"] = df_l4["主要工作任务"].astype(str).str.strip()
    df_l4["parent_code"] = df_l4["code"].apply(_get_parent_code)
    df_l4["l2_prefix"] = df_l4["code"].apply(_get_l2_prefix)
    df_l4["category"] = df_l4["code"].apply(_get_category)

    # 输出：保留大典原始列 + 衍生列
    output_cols = ["层级", "职业编码", "职业名称", "职业描述", "主要工作任务",
                   "parent_code", "l2_prefix", "category"]
    df_out = df_l4[output_cols].copy()
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # 统计
    print(f"\n📊 输出统计：")
    print(f"   文件：{output_csv}")
    print(f"   总节点：{len(df_out)}")
    cat_dist = df_out["category"].value_counts()
    for cat, cnt in cat_dist.items():
        print(f"     {cat}：{cnt} 个")
    
    # 统计主要工作任务覆盖率
    has_tasks = (df_out["主要工作任务"] != "").sum()
    print(f"\n   主要工作任务覆盖率：{has_tasks}/{len(df_out)} = {has_tasks/len(df_out)*100:.1f}%")

    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解析大典CSV，输出四级节点清单")
    parser.add_argument("--csv", type=str, default=DEFAULT_DICT_CSV,
                        help="大典CSV文件路径")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV,
                        help="输出CSV路径")
    args = parser.parse_args()

    load_and_filter(args.csv, args.output)
