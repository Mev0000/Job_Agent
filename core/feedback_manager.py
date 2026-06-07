"""
core/feedback_manager.py
Track 2 — 数据飞轮（Human-in-the-Loop）

功能：
1. ErrorCaseStore：存储叙事型纠错案例（用于 few-shot 注入）
2. GraphUpdateProposal：提议图谱口诀更新（需人工审核）
3. Few-Shot 动态组装：从 ErrorCaseStore 中抽取高频错误类型，注入 Prompt

设计依据：融合方案 v4.0 + P1-R（选项1：全量预留）+ P2-S（选项1：叙事型）
"""

import json
import os
import re
from datetime import datetime
from typing import List, Dict, Optional


# ============================================================
# 1. ErrorCaseStore — 叙事型纠错案例存储
# ============================================================

ERROR_CASE_PATH = "data/error_cases/error_case_store.jsonl"
MAX_STORE_SIZE = 2000  # 最多存储2000条，防止文件过大


def _ensure_dir():
    os.makedirs(os.path.dirname(ERROR_CASE_PATH), exist_ok=True)


def submit_correction(
    jd_text: str,
    wrong_code: str,
    correct_code: str,
    error_reason: str,
    correction_logic: str,
    confidence_at_time: int = 0,
    raw_reasoning: Optional[dict] = None,
) -> dict:
    """
    人工纠正后调用：写入一条叙事型纠错案例到 ErrorCaseStore。
    
    参数：
        jd_text: 原始 JD 文本
        wrong_code: 模型错误预测的代码
        correct_code: 人工标注的正确代码
        error_reason: 错误原因（拔高 / 99兜底滥用 / Step3走偏 / 交付物误判 / 其他）
        correction_logic: 纠偏逻辑（教会模型如何「悬崖勒马」的自然语言描述）
        confidence_at_time: 出错时的置信度评分
        raw_reasoning: 模型原始推理过程（可选，用于深度分析）
    
    返回：
        写入的记录字典
    """
    _ensure_dir()
    
    record = {
        "jd_text": jd_text[:2000],  # 截断，防止过大
        "wrong_code": wrong_code,
        "correct_code": correct_code,
        "error_reason": error_reason,
        "correction_logic": correction_logic,
        "confidence_at_time": confidence_at_time,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if raw_reasoning:
        record["raw_reasoning"] = json.dumps(raw_reasoning, ensure_ascii=False)[:1000]
    
    # 追加写入 .jsonl
    with open(ERROR_CASE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # 文件大小保护：超过 MAX_STORE_SIZE 条时，删除最早的 20%
    _trim_if_needed()
    
    return record


def _trim_if_needed():
    """超过最大条数时，删除最早的 20% 记录"""
    if not os.path.exists(ERROR_CASE_PATH):
        return
    with open(ERROR_CASE_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) <= MAX_STORE_SIZE:
        return
    keep_from = int(len(lines) * 0.2)
    with open(ERROR_CASE_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines[keep_from:])


def load_error_cases(n: int = 50) -> List[Dict]:
    """
    读取最近的 n 条纠错案例，用于 few-shot 动态注入。
    按时间戳降序排列（最新的在前）。
    """
    if not os.path.exists(ERROR_CASE_PATH):
        return []
    with open(ERROR_CASE_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    records = [json.loads(l) for l in lines if l.strip()]
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return records[:n]


def get_error_stats() -> Dict:
    """统计各类错误原因的分布，用于诊断模型弱点"""
    cases = load_error_cases(n=99999)
    stats = {}
    for c in cases:
        reason = c.get("error_reason", "未知")
        stats[reason] = stats.get(reason, 0) + 1
    return stats


# ============================================================
# 2. Few-Shot 动态组装 — 从 ErrorCaseStore 生成反思型 few-shot
# ============================================================

def build_fewshot_from_error_cases(
    error_type: str = "拔高",
    n: int = 2,
    exclude_codes: Optional[List[str]] = None,
) -> str:
    """
    从 ErrorCaseStore 中抽取指定错误类型的案例，组装成反思型 few-shot 文本。
    
    参数：
        error_type: 错误类型（拔高 / 99兜底滥用 / Step3走偏 / 交付物误判）
        n: 最多取几条
        exclude_codes: 排除某些代码（避免与现有 few-shot 重复）
    
    返回：
        可直接注入 Prompt 的 few-shot 文本（格式与 templates.py 中的示例一致）
    """
    cases = load_error_cases(n=99999)
    filtered = [
        c for c in cases
        if c.get("error_reason") == error_type
        and c.get("correction_logic")
        and (not exclude_codes or c.get("correct_code") not in exclude_codes)
    ]
    if not filtered:
        return ""
    
    snippets = []
    for i, case in enumerate(filtered[:n]):
        snippet = f"""
### 反思案例 #{i+1}（{error_type}错误）
- **JD文本**：{case['jd_text'][:300]}
- **模型错误预测**：`{case['wrong_code']}`
- **人工纠正为**：`{case['correct_code']}`
- **悬崖勒马逻辑**：{case['correction_logic']}
"""
        snippets.append(snippet)
    
    header = f"\n---\n## 历史反思案例（{error_type}类错误，自动注入）\n"
    return header + "\n".join(snippets)


def batch_prepare_fewshot(
    error_type_weights: Optional[Dict[str, int]] = None,
    total_budget: int = 3,
) -> str:
    """
    按错误类型权重，自动分配 few-shot 预算，生成动态 few-shot 注入文本。
    
    参数：
        error_type_weights: {错误类型: 权重}，默认均等
        total_budget: 最多注入几条 few-shot（防止 token 超限）
    
    返回：
        动态 few-shot 文本，直接拼接到 system_prompt 末尾
    """
    if error_type_weights is None:
        error_type_weights = {
            "拔高": 2,
            "99兜底滥用": 1,
            "Step3走偏": 1,
            "交付物误判": 1,
        }
    
    # 按权重分配预算
    types_sorted = sorted(error_type_weights.items(), key=lambda x: -x[1])
    results = []
    remaining = total_budget
    for etype, weight in types_sorted:
        take = min(weight, remaining)
        if take <= 0:
            break
        frag = build_fewshot_from_error_cases(etype, n=take)
        if frag:
            results.append(frag)
            remaining -= take
        if remaining <= 0:
            break
    
    if not results:
        return ""
    return "\n\n# 💡 动态注入：历史反思案例\n" + "\n".join(results)


# ============================================================
# 3. GraphUpdateProposal — 提议图谱口诀更新（人工审核后生效）
# ============================================================

PROPOSAL_PATH = "data/graph_proposals/proposals.jsonl"


def _ensure_proposal_dir():
    os.makedirs(os.path.dirname(PROPOSAL_PATH), exist_ok=True)


def propose_graph_update(
    source_code: str,
    target_code: str,
    proposed_rule: str,
    error_case_id: Optional[str] = None,
    proposer: str = "data_flywheel",
) -> dict:
    """
    基于 ErrorCase 提议新增/修改图谱口诀，写入 proposals.jsonl，等待人工审核。
    
    人工审核通过后，调用 graph_manager.update_expert_rule() 生效。
    
    参数：
        source_code: 易混淆的源头代码
        target_code: 目标代码
        proposed_rule: 提议的口诀文本
        error_case_id: 关联的 error_case 时间戳（用于溯源）
        proposer: 提议者（"data_flywheel" / "human_expert"）
    
    返回：
        提议记录字典
    """
    _ensure_proposal_dir()
    
    proposal = {
        "source_code": source_code,
        "target_code": target_code,
        "proposed_rule": proposed_rule,
        "error_case_id": error_case_id,
        "proposer": proposer,
        "status": "PENDING_REVIEW",  # PENDING_REVIEW / APPROVED / REJECTED
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reviewed_at": None,
        "reviewer": None,
    }
    
    with open(PROPOSAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(proposal, ensure_ascii=False) + "\n")
    
    return proposal


def list_pending_proposals() -> List[Dict]:
    """列出所有待审核的图谱更新提议"""
    if not os.path.exists(PROPOSAL_PATH):
        return []
    with open(PROPOSAL_PATH, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip() and json.loads(l).get("status") == "PENDING_REVIEW"]


def approve_proposal(timestamp: str, reviewer: str = "human_expert") -> dict:
    """
    人工审核通过提议，调用 graph_manager 生效。
    返回更新后的 proposal 记录。
    """
    import sys; sys.path.insert(0, os.path.dirname(__file__) + "/..")
    from core.graph_manager import create_graph_manager
    import yaml
    
    cfg_path = "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    proposals = []
    target = None
    with open(PROPOSAL_PATH, "r", encoding="utf-8") as f:
        for l in f:
            if not l.strip():
                continue
            p = json.loads(l)
            if p.get("created_at") == timestamp and p.get("status") == "PENDING_REVIEW":
                p["status"] = "APPROVED"
                p["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
                p["reviewer"] = reviewer
                target = p
            proposals.append(p)
    
    if not target:
        return {"error": "proposal not found or already reviewed"}
    
    # 调用 GraphManager 生效
    try:
        gm = create_graph_manager(config)
        gm.update_expert_rule(
            target["source_code"],
            target["target_code"],
            target["proposed_rule"],
        )
    except Exception as e:
        return {"error": f"GraphManager update failed: {e}"}
    
    # 回写文件
    with open(PROPOSAL_PATH, "w", encoding="utf-8") as f:
        for p in proposals:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    return target


# ============================================================
# 4. 与 StateMachine 集成 — 在 REVIEW_SUGGESTED 时自动写入 ErrorCaseStore
# ============================================================

def auto_collect_review_case(
    jd_text: str,
    predicted_code: str,
    confidence: int,
    reasoning: dict,
    auto_error_reason: str = "未分类",
) -> Optional[dict]:
    """
    当人工在 UI 中纠正 REVIEW_SUGGESTED 案例时调用。
    
    调用时机（由外部 UI/API 触发，非自动）：
    1. 系统输出 REVIEW_SUGGESTED，状态返回给前端
    2. 专家在 UI 中纠正为正确代码，并填写 error_reason 和 correction_logic
    3. 前端调用此函数（或等效 API），写入 ErrorCaseStore
    
    此函数为「纯数据写入层」，不包含 UI 逻辑。
    UI 层由平台团队另行开发（P1-R 待规划）。
    """
    # 此函数由人工审核 UI 调用，不自动触发
    # 保留接口定义，等待 UI 对接
    pass


# ============================================================
# 5. 诊断与可视化
# ============================================================

def print_error_stats():
    """打印错误类型分布，用于诊断模型弱点"""
    stats = get_error_stats()
    if not stats:
        print("📊 ErrorCaseStore 暂无数据")
        return
    print("📊 ErrorCaseStore 错误类型分布：")
    total = sum(stats.values())
    for reason, cnt in sorted(stats.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {reason:<20s}: {cnt:4d} 条 ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    # 单元测试：写入一条测试纠错案例
    test_record = submit_correction(
        jd_text="负责区域客户维护，执行总部营销方案，定期提交销售报表",
        wrong_code="2-02-10",
        correct_code="4-01-02",
        error_reason="拔高",
        correction_logic="JD中无「产品架构/研发统筹」等创制性动词，只有「执行/提交报表」等执行类动作，不可判入2类。应严格按 Level 1.3 口诀：代码含拔高词且JD无对应创制动作 → 默认4类。",
        confidence_at_time=55,
    )
    print("✅ 测试记录已写入 ErrorCaseStore：")
    print(json.dumps(test_record, ensure_ascii=False, indent=2))
    
    print("\n📊 当前错误统计：")
    print_error_stats()
    
    print("\n🔍 动态 few-shot 注入测试：")
    fewshot = batch_prepare_fewshot(total_budget=2)
    print(fewshot[:500] if fewshot else "（无数据）")
