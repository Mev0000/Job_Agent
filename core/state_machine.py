# core/state_machine.py

import json
import re
import traceback
from typing import Dict, List, Optional

class JobAgentStateMachine:
    def __init__(self, llm_client, retriever, config, rules_prompt, json_template):
        """
        初始化 Agent 状态机
        :param llm_client: Gemma4Client 实例
        :param retriever: 双轨融合引擎实例 (内部自带 graph_rag 与 corpus)
        """
        self.llm = llm_client
        self.retriever = retriever          
        self.graph_rag = retriever.graph    
        self.max_hops = config.get("agent", {}).get("max_hops", 6)
        
        self.system_prompt = f"{rules_prompt}\n\n{json_template}"
        self.global_dict_tree = getattr(self.graph_rag, 'global_dict_tree', {})
        
        # 置信度评分配置（可通过config.yaml调整）
        self.confidence_config = config.get("agent", {}).get("confidence", {
            "threshold_review": 65,  # 低于此分数标记为REVIEW_SUGGESTED
            "score_per_dimension": 5,  # 每个维度一致+5分（6维共30分满额）
            "score_rule_hit": 10,       # 口诀强制命中加分
            "score_boundary_check": 5,   # 边界预检(check_0)有正向指引加分
            "penalty_99": -15,          # 使用99兜底减分（下调，区分合法穷举与滥用）
            "penalty_p1_mismatch": -15,  # P1不一致但仍选择减分
            "penalty_global_search": -10 # 经过GLOBAL_SEARCH重定向减分
        })

    def _extract_json_from_text(self, text):
        """增强版 JSON 提取器，防 Markdown 干扰与非法转义"""
        # 🌟 修复 1：取消截断，打印全量日志以便排查！
        print(f"\n[🚨 DEBUG 监控] 模型输出 (全量展示)：\n{text}\n{'='*40}")
        
        # 🌟 修复 2：预处理字符串值内的实际换行符等控制字符
        # LLM 有时会在 JSON 字符串值中直接输出实际换行符（而非 \n 转义），
        # 这会导致 json.loads 报 "Expecting ',' delimiter" 错误。
        # 用状态机精确转义字符串值内的控制字符，不影响字符串外的格式。
        def _escape_control_chars(s: str) -> str:
            result = []
            in_string = False
            escape_next = False
            for c in s:
                if escape_next:
                    # 如果在字符串内部且下一个字符不是合法的 JSON 转义字符
                    # （如 \approx 中的 \a），则多输出一个反斜杠将其转义
                    if in_string and c not in '"\\/bfnrtu':
                        result.append('\\')
                    result.append(c)
                    escape_next = False
                elif c == '\\':
                    result.append(c)
                    escape_next = True
                elif c == '"':
                    if not escape_next:
                        in_string = not in_string
                    result.append(c)
                elif in_string:
                    if c == '\n':
                        result.append('\\n')
                    elif c == '\r':
                        result.append('\\r')
                    elif c == '\t':
                        result.append('\\t')
                    else:
                        result.append(c)
                else:
                    result.append(c)
            # 如果字符串以孤立的反斜杠结尾，补一个转义
            if escape_next and in_string:
                result.append('\\')
            return ''.join(result)
        
        try:
            clean_text = _escape_control_chars(text)
            
            # 策略 1：优先匹配被 ```json ... ``` 包裹的内容
            match = re.search(r'```json\s*(.*?)\s*```', clean_text, re.DOTALL)
            if match:
                return json.loads(match.group(1).strip(), strict=False)
                
            # 策略 2：如果没有包裹，寻找最外层的 {}
            match = re.search(r'(\{.*\})', clean_text, re.DOTALL)
            if match:
                return json.loads(match.group(1).strip(), strict=False)
                
            return None
        except Exception as e:
            print(f"⚠️ 模型输出解析 JSON 失败: {e}")
            return None

    def _calculate_confidence(self, result_json: Dict, reasoning: Dict, current_hop: int,
                              trajectory_log: Optional[List[str]] = None,
                              is_rule_forced: bool = False) -> int:
        """
        根据客观特征计算置信度评分（非LLM自评，防止过度自信）
        
        评分规则（v3.5 梯度化重设计）：
        基础分: 50
        +5/维度: 七维比对中每1维"一致" +5分（最多6维=+30）
        +10分: 口诀强制命中（is_rule_forced=True）
        +5分: check_0 边界预检有正向指引（非"无边界命中"）
        -15分: 使用99兜底代码
        -15分: ≥2维明确不一致但仍选择该候选
        -10分: 经过GLOBAL_SEARCH重定向
        
        分类规则：与提示词中的"七维中≥5维一致 → 高置信度 FINALIZE"保持一致

        返回：0-100的整数置信度评分
        """
        score = 50  # 基础分
        
        # 获取配置
        cfg = self.confidence_config
        score_per_dim = cfg.get("score_per_dimension", 5)
        score_rule = cfg.get("score_rule_hit", 10)
        score_boundary = cfg.get("score_boundary_check", 5)
        penalty_99 = cfg.get("penalty_99", -15)
        penalty_p1 = cfg.get("penalty_p1_mismatch", -15)
        penalty_gs = cfg.get("penalty_global_search", -10)
        
        # ── 检查是否使用99兜底 ──
        result_code = result_json.get("result_code", "")
        if result_code.endswith("99"):
            score += penalty_99
        
        # ── P1-B: GLOBAL_SEARCH 轨迹检测 ──
        if trajectory_log and any("GLOBAL_SEARCH" in step for step in trajectory_log):
            score += penalty_gs
        elif trajectory_log is None and current_hop > 3:
            score += penalty_gs
            
        # ── P1-A: 口诀强制命中 ──
        if is_rule_forced:
            score += score_rule
            
        # ── 七维比对梯度评分（v3.5: 从"≥5全匹配"改为按维度梯度计分）──
        # v3.3同步：key 名为 step_3_evidence_cross_match
        cross_match = reasoning.get("step_3_evidence_cross_match", {})
        # 兼容旧版 v3.2 key 名
        if not cross_match:
            cross_match = reasoning.get("step_4_evidence_cross_match", {})
        
        if isinstance(cross_match, dict):
            check_keys = [
                "check_1_deliverables",
                "check_2_core_actions", 
                "check_3_objects",
                "check_4_environment",
                "check_5_role_level",
                "check_6_served_population"
            ]
            checks = [str(cross_match.get(k, "")) for k in check_keys]
            
            # 🔧 修复子串冲突："不一致" 中 "一致" 是子串，需先排除否定后再判正向
            negative_kw = {"不一致", "不匹配", "不符合", "不吻合", "不契合"}
            positive_kw = {"一致", "匹配", "符合", "吻合", "契合"}
            
            def _is_negative(text: str) -> bool:
                return any(kw in text for kw in negative_kw)
            
            def _is_positive(text: str) -> bool:
                # 先排除否定（避免"不一致"子串误匹配"一致"）
                if _is_negative(text):
                    return False
                return any(kw in text for kw in positive_kw)
            
            # 梯度加分：每个正向维度 +score_per_dim
            positive_count = sum(1 for c in checks if _is_positive(c))
            score += positive_count * score_per_dim
            
            # 不一致惩罚：≥2维明确不一致
            negative_count = sum(1 for c in checks if _is_negative(c))
            if negative_count >= 2:
                score += penalty_p1
                print(f"  ⚠️ 置信度扣分：前6项有{negative_count}项不一致仍选择 ({penalty_p1}分)")
        
        # ── check_0 边界预检贡献 ──
        check_0 = str(reasoning.get("check_0_boundary_precheck", ""))
        # v3.3兼容：check_0 可能在 step_3 内部
        if not check_0 and isinstance(cross_match, dict):
            check_0 = str(cross_match.get("check_0_boundary_precheck", ""))
        if check_0 and not any(kw in check_0 for kw in ["无边界命中", "未命中"]):
            score += score_boundary

        # 确保分数在0-100范围内
        score = max(0, min(100, score))
        return score

    def _audit_hop_structure(self, current_hop: int, reasoning: Dict, result_code: str,
                             exploration_ledger: List[str]) -> Dict:
        """
        【P1 跳数结构审计】分析为什么跳数增多时准确率下降。

        度量维度：
        1. 方向漂移度：当前跳的 L2 大类与历史路径是否一致
        2. 证据收敛度：Step 4 七维比对中"一致"的维度数（0-6）
        3. 假说翻转度：Step 3 是否推翻了上一跳的结论

        Returns: 审计 dict，含 drift_count, convergence_7d, reversal_flag, risk_level
        """
        # 🔧 修复: result_code 可能是 list（如 REQ_L2 时），需先转 str
        if isinstance(result_code, list):
            result_code_str = result_code[0] if result_code else ""
        else:
            result_code_str = result_code or ""
        current_l2 = "-".join(result_code_str.split("-")[:2]) if result_code_str else "UNKNOWN"

        # 1. 方向漂移检测
        import re as _re
        historical_l2 = []
        for entry in exploration_ledger:
            codes = _re.findall(r'\(([^)]+)\)', entry)
            if codes:
                l2 = "-".join(codes[0].split("-")[:2])
                historical_l2.append(l2)

        drift_count = 0
        if len(historical_l2) >= 2:
            for i in range(1, len(historical_l2)):
                if historical_l2[i] != historical_l2[i-1]:
                    drift_count += 1

        # 2. 证据收敛度（v3.3同步：key 为 step_3_evidence_cross_match）
        cross_match = reasoning.get("step_3_evidence_cross_match", {})
        if not cross_match:
            cross_match = reasoning.get("step_4_evidence_cross_match", {})
        convergence = 0
        positive_kw = {"一致", "匹配", "符合", "吻合", "契合"}
        negative_kw = {"不一致", "不匹配", "不符合", "不吻合", "不契合"}
        if isinstance(cross_match, dict):
            for ck in ["check_1_deliverables", "check_2_core_actions", "check_3_objects",
                       "check_4_environment", "check_5_role_level", "check_6_served_population"]:
                text = str(cross_match.get(ck, ""))
                # 先排除否定再判正向（修复子串冲突）
                if not any(kw in text for kw in negative_kw):
                    if any(kw in text for kw in positive_kw):
                        convergence += 1

        # 3. 假说翻转检测（v3.3同步：key 为 step_2_hypothesis）
        hypothesis = str(reasoning.get("step_2_hypothesis", ""))
        if not hypothesis:
            hypothesis = str(reasoning.get("step_3_substance_hypothesis", ""))
        reversal_flag = any(kw in hypothesis for kw in {"纠正", "撤回", "推翻", "重新定位", "改判"})

        # 风险评估
        risks = []
        if drift_count >= 2:
            risks.append(f"方向漂移{drift_count}次—LLM在多个大类间反复跳跃")
        if convergence <= 2 and current_hop >= 3:
            risks.append(f"跳{current_hop}但证据收敛度仅{convergence}/6—候选池与JD整体不匹配")
        if reversal_flag and current_hop >= 3:
            risks.append(f"跳{current_hop}出现假说翻转—此前推理链存在根本性错误")

        audit = {
            "hop": current_hop, "current_l2": current_l2,
            "drift_count": drift_count, "convergence_7d": convergence,
            "hypothesis_reversal": reversal_flag,
            "risk_level": "HIGH" if len(risks) >= 2 else ("MEDIUM" if len(risks) == 1 else "LOW"),
            "risks": risks
        }
        return audit
            
    def _build_compare_table(self, compare_codes: list, job_name: str, job_desc: str) -> str:
        """P3: 构建并排对比表 — 为 COMPARE 动作提供 2-3 个候选的 7D 属性 + 混淆口诀对比"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"📋 岗位: {job_name}")
        lines.append(f"📝 JD摘要: {job_desc[:120]}...")
        lines.append("=" * 70)

        # 收集每个代码的特征
        code_features = {}
        for code in compare_codes:
            features = {}
            # 尝试从 graph 获取 7D 特征
            try:
                node_data = self.retriever.graph.nodes.get(code, {})
                features["name"] = node_data.get("name", node_data.get("职业名称", code))
                features["core_actions"] = node_data.get("core_actions", node_data.get("动作", []))
                features["deliverables"] = node_data.get("deliverables", node_data.get("交付物", []))
                features["served_population"] = node_data.get("served_population", node_data.get("服务对象", ""))
                features["environment"] = node_data.get("environment", node_data.get("工作环境", ""))
                features["role_level"] = node_data.get("role_level", node_data.get("职业层级", ""))
                features["is_gov"] = node_data.get("is_government", node_data.get("是否涉公权", False))
                features["is_clinical"] = node_data.get("is_clinical", node_data.get("是否涉临床", False))
            except Exception:
                pass
            # 回退：从 corpus 中查找
            if not features.get("name") or features["name"] == code:
                for item in self.retriever.corpus:
                    if item["code"] == code:
                        features["name"] = item.get("text", code)[:80]
                        break
            code_features[code] = features

        # 构建并排对比
        for i, code in enumerate(compare_codes, 1):
            f = code_features.get(code, {})
            name = f.get("name", code)
            lines.append(f"\n{'─' * 35} 候选 {i}: [{code}] {name} {'─' * 35}")
            
            actions = f.get("core_actions", [])
            actions_str = ", ".join(actions[:5]) if actions else "（无数据）"
            lines.append(f"  🔧 核心动作: {actions_str}")
            
            deliverables = f.get("deliverables", [])
            deliv_str = ", ".join(deliverables[:5]) if deliverables else "（无数据）"
            lines.append(f"  📦 交付物:   {deliv_str}")
            
            env = f.get("environment", "（无数据）")
            lines.append(f"  🏠 工作环境: {str(env)[:80]}")
            
            sp = f.get("served_population", "（无数据）")
            lines.append(f"  👥 服务对象: {str(sp)[:60]}")
            
            rl = f.get("role_level", "（无数据）")
            lines.append(f"  🏷️ 职业层级: {str(rl)[:40]}")
            
            # 法理红线
            red_flags = []
            if f.get("is_gov"): red_flags.append("⚠️涉公权")
            if f.get("is_clinical"): red_flags.append("⚠️涉临床")
            if red_flags:
                lines.append(f"  ⚖️ 法理红线: {' | '.join(red_flags)}")

        # 混淆口诀（两两对比）
        lines.append(f"\n{'═' * 70}")
        lines.append("🚨 【混淆口诀 — 两两鉴别诊断】")
        lines.append(f"{'═' * 70}")
        has_rule = False
        for i in range(len(compare_codes)):
            for j in range(i + 1, len(compare_codes)):
                a, b = compare_codes[i], compare_codes[j]
                rule = None
                try:
                    rule = self.retriever.graph.get_confusion_text(a, b)
                except Exception:
                    pass
                if not rule:
                    try:
                        rule = self.retriever.graph.get_confusion_text(b, a)
                    except Exception:
                        pass
                if rule:
                    has_rule = True
                    lines.append(f"\n  [{a}] vs [{b}]:")
                    lines.append(f"  👉 {rule}")
        if not has_rule:
            lines.append("  （无预存混淆口诀，请基于上方7D属性自行比对）")

        lines.append(f"\n{'═' * 70}")
        return "\n".join(lines)

    def run(self, job_name, job_desc, initial_candidates_context):
        """
        运行状态机主循环
        """

        try:
            current_hop = 1
        
            # 将最原始的 JD 和初筛库封装为基石 Prompt，防止后续跳跃中失忆
            base_user_prompt = f"【目标岗位】\n名称：{job_name}\n职责：{job_desc}\n\n【初筛参考库 (Hop-1)】\n{initial_candidates_context}"
            current_user_prompt = base_user_prompt
        
            trajectory_log = []
            exploration_ledger = []
            _confusion_checked_codes = set()  # P3: 防止口诀二次校验死循环
            _compared_codes = set()  # P3-bis: 防止自动COMPARE死循环

            while current_hop <= self.max_hops:
                print(f"  ➡️ 开始第 {current_hop} 跳推理...")
            
                # 1. 调用大模型
                response_text = self.llm.generate(self.system_prompt, current_user_prompt)
            
                if not response_text:
                    trajectory_log.append("=== 模型无响应 ===")
                    # 🔧 修复：无响应时不直接 raise，改为跳过本跳（与格式错误处理一致）
                    # 允许最多重试2次，超过则终止
                    current_hop += 1
                    if current_hop > self.max_hops:
                        raise RuntimeError("模型无响应，已终止推理")
                    print(f"  ⚠️ 模型无响应，跳过本跳，继续推理（已重试）...")
                    continue
                
                trajectory_log.append(f"=== Hop {current_hop} 模型输出 ===\n{response_text}\n")
            
                # 2. 解析动作
                result_json = self._extract_json_from_text(response_text)
            
                if not result_json or 'action' not in result_json:
                    print("  ❌ 未能解析出标准 JSON 或缺少 action 字段，尝试重试。")
                    current_hop += 1
                    continue
                
                action = result_json.get('action')
                result_code = result_json.get('result_code', 'UNKNOWN')
                reasoning = result_json.get("reasoning", {})
            
                # P1-A: 从候选字典中提取 is_rule_forced 标记
                # 该标记由 retriever.retrieve() 在找到图谱口诀强制匹配时写入候选字典
                # 格式示例：{"is_rule_forced": true, "forced_code": "4-01-02"}
                is_rule_forced = result_json.get("is_rule_forced", False)
            
                # 🔬 P1: 跳数结构审计（研究为什么跳数多了准确率下降，而非简单限制跳数）
                hop_audit = self._audit_hop_structure(
                    current_hop, reasoning, result_code, exploration_ledger
                )
                trajectory_log.append(
                    f"=== Hop {current_hop} 结构审计 ===\n"
                    f"L2={hop_audit['current_l2']} | 漂移={hop_audit['drift_count']}次 | "
                    f"7D收敛={hop_audit['convergence_7d']}/6 | 翻转={hop_audit['hypothesis_reversal']} | "
                    f"风险={hop_audit['risk_level']}\n"
                    + (f"风险详情: {'; '.join(hop_audit['risks'])}\n" if hop_audit['risks'] else "")
                )
            
                step_summary = reasoning.get("step_4_final_decision", "未完成定谳")
                if not step_summary or step_summary == "未完成定谳":
                    step_summary = reasoning.get("step_5_isomorphic_finalize", "未完成定谳")
                # 压缩结论为关键词格式（节省Token）
                if action == "GLOBAL_SEARCH":
                    brief = "候选集失效"
                elif action == "REQ_L2":
                    brief = f"锁定{result_code}"
                elif action == "FINALIZE":
                    brief = f"定谳:{result_code}"
                elif action == "REQ_L3_FULL":
                    brief = f"深挖:{result_code}"
                elif action == "COMPARE":
                    brief = f"对比:{result_code}"
                elif action == "8-00-00":
                    brief = "空壳岗位"
                else:
                    brief = step_summary[:15] if step_summary else "未知"
                
                exploration_ledger.append(f"[{current_hop}]{action}({result_code}):{brief}")
            
                # ---------------------------------------------
                # 分支 1：成功定谳
                # ---------------------------------------------
                if action == "FINALIZE":
                    # 🔧 修复：截断四级代码为三级（如 2-02-10-06 → 2-02-10）
                    import re as _re
                    if isinstance(result_code, str):
                        _trimmed = _re.sub(r'^(\d-\d{2}-\d{2})-\d+$', r'\1', result_code.strip())
                        if _trimmed != result_code:
                            print(f"  ✂️ [Hop {current_hop}] 四级代码截断: {result_code} → {_trimmed}")
                            result_code = _trimmed

                    # ── P1: 审计门控（在 FINALIZE 返回前执行）──
                    convergence = hop_audit['convergence_7d']
                    drift_count = hop_audit['drift_count']
                    risk_level = hop_audit['risk_level']
                    
                    # 层1: Hop-1 收敛度过低 → 强制禁止 FINALIZE，导向 REQ_L2
                    if current_hop == 1 and convergence <= 2:
                        print(f"  🚫 [Hop-1 审计门控] 收敛度仅 {convergence}/6 ≤2，候选池与JD严重不匹配！")
                        print(f"  🚫 强制拦截 FINALIZE，导向 REQ_L2 调阅大类全景。")
                        trajectory_log.append(
                            f"=== Hop-1 审计干预 ===\n"
                            f"收敛度 {convergence}/6 ≤2，候选池质量可疑，强制改为 REQ_L2。\n"
                        )
                        best_l2 = "-".join(result_code.split("-")[:2]) if result_code else "2-02"
                        if best_l2.endswith("-99"):
                            best_l2 = "4-01"  # 最安全的大类兜底
                        # 手动构造 REQ_L2 memory_block 并 continue
                        requested_l2_codes = [best_l2]
                        l2_features = []
                        l3_codes_list = []
                        for l2_code in requested_l2_codes:
                            for item in self.retriever.corpus:
                                if item['code'].startswith(l2_code):
                                    if re.match(r'^\d-\d{2}-\d{2}$', item['code']):
                                        clean_text = re.sub(r'主要工作任务.*', '', item['text'], flags=re.DOTALL).strip()
                                        l3_codes_list.append(f"[{item['code']}] {clean_text}")
                        l2_features.extend(l3_codes_list)
                        if l3_codes_list:
                            l2_features.append("")
                            l2_features.append("【🚨 提示】：以上只显示三级代码。如需查看四级细节，请输出 REQ_L3_FULL！")
                        full_dict_context = "\n".join(l2_features) if l2_features else f"⚠️ 未找到大类 {requested_l2_codes}。"
                        ledger_text = "\n".join(exploration_ledger) if exploration_ledger else "无"
                        scoped_recall_context = ""
                        try:
                            scoped_recall_context = self.retriever.retrieve_scoped(
                                job_name, job_desc, requested_l2_codes
                            )
                        except Exception as e:
                            print(f"  ⚠️ [审计-二次召回] 异常: {e}")
                        memory_block = f"""【🧭 全局探索轨迹账本】
    {ledger_text}

    🚫🚫🚫 【系统审计强制干预 — Hop-1 FINALIZE 被拦截】 🚫🚫🚫
    你上一跳试图 FINALIZE({result_code})，被审计门控拦截！
    原因：七维证据收敛度仅 {convergence}/6 ≤2，候选池与JD整体匹配度过低。

    {f"【🎯 二次向量精准召回】{scoped_recall_context}" if scoped_recall_context else ""}

    【大类全景定义（{best_l2}）】
    {full_dict_context}

    🎯 【强制指令】：
    1. 先检查二次向量召回候选（如有），做七维交叉验证
    2. 若候选匹配 → FINALIZE
    3. 若都不匹配但定义列表有合适的 → FINALIZE
    4. 若确认此大类也不对 → REQ_L2 换路
    5. 🚨 彻底迷失 → GLOBAL_SEARCH"""
                        current_user_prompt = f"{base_user_prompt}\n\n======================\n{memory_block}"
                        current_hop += 1
                        continue  # 重要：跳过 FINALIZE return，进入下一跳

                    print(f"  ✅ [Hop {current_hop}] 成功定谳，代码: {result_code}")
                
                    # 计算置信度评分
                    confidence = self._calculate_confidence(
                        result_json, reasoning, current_hop,
                        trajectory_log=trajectory_log,
                        is_rule_forced=is_rule_forced
                    )
                    
                    # 层2: 审计风险等级 HIGH → 额外惩罚
                    if risk_level == "HIGH":
                        print(f"  🔴 [审计门控] 跳{current_hop} 风险等级 HIGH: {'; '.join(hop_audit['risks'])}")
                        confidence = max(confidence - 25, 0)
                        trajectory_log.append(
                            f"=== 审计高风险干预 ===\n"
                            f"风险详情: {'; '.join(hop_audit['risks'])}\n"
                            f"置信度额外 -25 分（当前: {confidence}）\n"
                        )
                    
                    # 层3: 方向漂移 + 低收敛 → 重度惩罚
                    if drift_count >= 2 and convergence <= 2:
                        print(f"  🔴 [审计门控] 方向漂移{drift_count}次 + 收敛度仅{convergence}/6，推理链极不稳定！")
                        confidence = max(confidence - 30, 0)
                        trajectory_log.append(
                            f"=== 审计不稳定干预 ===\n"
                            f"漂移{drift_count}次 + 收敛度{convergence}/6，置信度额外 -30 分\n"
                        )
                    
                    # Hop-1 预检门控（收敛度=3时触发警告，≤2已在层1拦截）
                    if current_hop == 1 and convergence == 3:
                        print(f"  ⚠️ [Hop-1 预检] 收敛度仅 {convergence}/6，定谳置信度降低")
                        confidence = max(confidence - 20, 0)
                        trajectory_log.append(
                            f"=== Hop-1 预检警告 ===\n"
                            f"收敛度仅 {convergence}/6，定谳可能不稳定。\n"
                        )
                
                    # ── P3: FINALIZE 口诀二次校验（混淆图谱最后一关）──
                    confused_opponents = []
                    confusion_texts = []
                    try:
                        confused_opponents = self.retriever.graph.get_confused_codes(result_code)
                        for opp in confused_opponents:  # 显示全部混淆对手，不设截断
                            if len(confusion_texts) >= 6:  # 安全上限6个（实际最多4-5个）
                                break
                            rule = self.retriever.graph.get_confusion_text(result_code, opp)
                            if not rule:
                                rule = self.retriever.graph.get_confusion_text(opp, result_code)
                            if rule:
                                confusion_texts.append(f"[{result_code}] vs [{opp}]: {rule}")
                    except Exception:
                        pass
                    
                    if confusion_texts:
                        trajectory_log.append(
                            f"=== FINALIZE 口诀二次校验 ===\n"
                            + "\n".join(confusion_texts) + "\n"
                        )
                        # 置信度<85且有混淆对手时，一律触发口诀二次校验（不再设下限）
                        if (confidence < 85
                            and result_code not in _confusion_checked_codes
                            and current_hop < self.max_hops):
                            _confusion_checked_codes.add(result_code)
                            print(f"  🔬 [口诀二次校验] 定谳 {result_code} 存在 {len(confused_opponents)} 个易混淆对手")
                            print(f"  🔬 置信度 {confidence} 处于边际区间，触发口诀复核...")
                            # 构造复核记忆块
                            confusion_block = "\n".join(confusion_texts)
                            verify_prompt = f"""【🔬 口诀最终复核 — 混淆图谱预警】
    你选择的定谳代码 [{result_code}] 在混淆图谱中与以下代码存在易混淆关系：
    {confusion_block}

    请基于以下口诀重新审视你的定谳：
    1. 你的 step_0 劳动事实是否符合 [{result_code}] 的定义（而非其对手）？
    2. 若确认正确 → 再次 FINALIZE 相同代码（confidence_score 不变）
    3. 若发现错误 → FINALIZE 更正后的代码（confidence_score 降低10-20分）
    4. 若无法抉择 → FINALIZE 你认为最可能的代码，但 confidence_score 降低15分

    【⚡ 必须 FINALIZE，这是最后一跳！】"""
                            current_user_prompt = f"{base_user_prompt}\n\n======================\n{verify_prompt}"
                            current_hop += 1
                            continue  # 重新进入循环，模型做最终确认
                
                    # ── P3-bis: 自动COMPARE触发（系统级拦截）──
                    if result_code and isinstance(result_code, str) and current_hop < self.max_hops:
                        if result_code not in _compared_codes:
                            try:
                                confused_ops = self.retriever.graph.get_confused_codes(result_code)
                                if confused_ops:
                                    _compared_codes.add(result_code)
                                    compare_codes = [result_code] + [op for op in confused_ops if op != result_code][:2]
                                    compare_table = self._build_compare_table(compare_codes, job_name, job_desc)
                                    print(f"  🔬 [自动COMPARE] 定谳 {result_code} 存在混淆对手，注入对比表...")
                                    compare_prompt = (
                                        "【🔬 系统自动触发：混淆代码对比】\n"
                                        f"你即将 FINALIZE 的代码 [{result_code}] 在混淆图谱中与以下代码存在易混淆关系：\n\n"
                                        f"{compare_table}\n\n"
                                        "请基于上方并排对比信息重新审视你的定谳：\n"
                                        f"1. 你的 step_0 劳动事实更符合哪个代码的定义？\n"
                                        f"2. 若确认 [{result_code}] 正确 → 再次 FINALIZE 相同代码\n"
                                        "3. 若发现错误 → FINALIZE 更正后的代码\n"
                                        "4. 若无法抉择 → FINALIZE 你认为最可能的代码，但 confidence_score 降低10-15分\n\n"
                                        "【⚡ 必须 FINALIZE，这是对比后的最后一跳！】"
                                    )
                                    current_user_prompt = f"{base_user_prompt}\n\n=====================\n{compare_prompt}"
                                    current_hop += 1
                                    continue
                            except Exception as e:
                                print(f"  ⚠️ [自动COMPARE] 异常: {e}")
                    
                    # 根据置信度决定状态
                    threshold = self.confidence_config.get("threshold_review", 65)
                    if confidence >= threshold:
                        status = "SUCCESS"
                    else:
                        status = "REVIEW_SUGGESTED"  # 建议人工复核
                
                    return {
                        "status": status, 
                        "code": result_code, 
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "log": trajectory_log
                    }
                
                # ---------------------------------------------
                # 分支 2：垃圾岗位熔断
                # ---------------------------------------------
                elif action == "8-00-00":
                    print(f"  🗑️ [Hop {current_hop}] 判定为空壳/垃圾岗位。")
                    return {
                        "status": "GARBAGE", 
                        "code": "8-00-00", 
                        "confidence": 0,  # 垃圾岗位置信度默认为0
                        "reasoning": reasoning,
                        "log": trajectory_log
                    }

                # ---------------------------------------------
                # 分支 3： 请求全局大纲 
                # ---------------------------------------------
                elif action == "GLOBAL_SEARCH":
                    print(f"  🚨 [Hop {current_hop}] 触发【上帝视角】：向量召回彻底失效，动态下发 2022 修订版全量一二级导航网...")

                    if current_hop >= self.max_hops:
                        print(f"  💥 [Hop {current_hop}] 达到最大跳数限制，熔断退出。转人工复核。")
                        return {"status": "MELTDOWN", "code": "MANUAL_REVIEW", "confidence": 0, "reasoning": reasoning, "log": trajectory_log}
                
                    # 完美对齐 2022 修订版大典官方最精准的一二级大类标准名称网
                    # 完美对齐 2022 修订版大典官方最精准的一二级大类标准名称网 (100%全量无损版)
                    macro_structure = {
                        "1": {
                            "name": "党的机关、国家机关、群众团体和社会组织、企事业单位负责人",
                            "definition": "在党政机关、企事业单位、社会团体、基层群众自治组织中，担任领导职务并具有决策、管理职权的人员。",
                            "all_l2": {
                                "1-01": "中国共产党机关负责人",
                                "1-02": "国家机关负责人",
                                "1-03": "民主党派和工商联负责人",
                                "1-04": "人民团体和群众团体、社会组织及其他成员组织负责人",
                                "1-05": "基层群众自治组织负责人",
                                "1-06": "企事业单位负责人"
                            }
                        },
                        "2": {
                            "name": "专业技术人员",
                            "definition": "从事科学研究、技术开发、设计、试验、检验、分析、运用和维护科学技术知识，以及从事专业技术工作的人员。",
                            "all_l2": {
                                "2-01": "科学研究人员",
                                "2-02": "工程技术人员",
                                "2-03": "农业技术人员",
                                "2-04": "飞机和船舶技术人员",
                                "2-05": "卫生专业技术人员",
                                "2-06": "经济和金融专业人员",
                                "2-07": "监察、法律、社会和宗教专业人员",
                                "2-08": "教学人员",
                                "2-09": "文学艺术、体育专业人员",
                                "2-10": "新闻出版、文化专业人员",
                                "2-99": "其他专业技术人员"
                            }
                        },
                        "3": {
                            "name": "办事人员和有关人员",
                            "definition": "在国家机关、企事业单位、社会团体、民办非企业单位中，从事行政业务、行政事务、经济业务、经济事务、警务辅助、安全保卫等办事工作的人员。",
                            "all_l2": {
                                "3-01": "行政办事及辅助人员",
                                "3-02": "安全和消防及辅助人员",
                                "3-03": "法律事务及辅助人员",
                                "3-99": "其他办事人员和有关人员"
                            }
                        },
                        "4": {
                            "name": "社会生产服务和生活服务人员",
                            "definition": "从事社会生产和生活服务、经营性服务、社会服务等工作的人员。",
                            "all_l2": {
                                "4-01": "批发与零售服务人员",
                                "4-02": "交通运输、仓储物流和邮政业服务人员",
                                "4-03": "住宿和餐饮服务人员",
                                "4-04": "信息传输、软件和信息技术服务人员",
                                "4-05": "金融服务人员",
                                "4-06": "房地产服务人员",
                                "4-07": "租赁和商务服务人员",
                                "4-08": "技术辅助服务人员",
                                "4-09": "水利、环境和公共设施管理服务人员",
                                "4-10": "居民服务人员",
                                "4-11": "电力、燃气及水供应服务人员",
                                "4-12": "修理及制作服务人员",
                                "4-13": "文化和教育服务人员",
                                "4-14": "健康、体育和休闲服务人员",
                                "4-99": "其他社会生产服务和生活服务人员"
                            }
                        },
                        "5": {
                            "name": "农、林、牧、渔业生产及辅助人员",
                            "definition": "从事农、林、牧、渔业种养殖、生产、加工等生产及辅助工作的人员。",
                            "all_l2": {
                                "5-01": "农业生产人员",
                                "5-02": "林业生产人员",
                                "5-03": "畜牧业生产人员",
                                "5-04": "渔业生产人员",
                                "5-05": "农、林、牧、渔业生产辅助人员",
                                "5-99": "其他农、林、牧、渔业生产及辅助人员"
                            }
                        },
                        "6": {
                            "name": "生产制造及有关人员",
                            "definition": "从事矿产开采、产品制造、工程施工、加工、组装、检测、设备操作等生产制造及相关工作的人员。",
                            "all_l2": {
                                "6-01": "农副产品加工人员",
                                "6-02": "食品、饮料生产加工人员",
                                "6-03": "烟草及其制品加工人员",
                                "6-04": "纺织、针织、印染人员",
                                "6-05": "纺织品、服装和皮革、毛皮制品加工制作人员",
                                "6-06": "木材加工、家具与木制品制作人员",
                                "6-07": "纸及纸制品生产加工人员",
                                "6-08": "印刷和记录媒介复制人员",
                                "6-09": "文教、工美、体育和娱乐用品制造人员",
                                "6-10": "石油加工和炼焦、煤化工生产人员",
                                "6-11": "化学原料和化学制品制造人员",
                                "6-12": "医药制造人员",
                                "6-13": "化学纤维制造人员",
                                "6-14": "橡胶和塑料制品制造人员",
                                "6-15": "非金属矿物制品制造人员",
                                "6-16": "采矿人员",
                                "6-17": "金属冶炼和压延加工人员",
                                "6-18": "机械制造基础加工人员",
                                "6-19": "金属制品制造人员",
                                "6-20": "通用设备制造人员",
                                "6-21": "专用设备制造人员",
                                "6-22": "汽车制造人员",
                                "6-23": "铁路、船舶、航空设备制造人员",
                                "6-24": "电气机械和器材制造人员",
                                "6-25": "计算机、通信和其他电子设备制造人员",
                                "6-26": "仪器仪表制造人员",
                                "6-27": "再生资源综合利用人员",
                                "6-28": "电力、热力、气体、水生产和输配人员",
                                "6-29": "建筑施工人员",
                                "6-30": "运输设备和通用工程机械操作人员及有关人员",
                                "6-31": "生产辅助人员",
                                "6-99": "其他生产制造及有关人员"
                            }
                        },
                        "7": {
                            "name": "军队人员",
                            "definition": "在中国人民解放军、中国人民武装警察部队中服现役的军官、士官、义务兵等人员。",
                            "all_l2": {
                                "7-01": "军官（警官）",
                                "7-02": "军士（警士）",
                                "7-03": "义务兵",
                                "7-04": "文职人员"
                            }
                        },
                        "8": {
                            "name": "不便分类的其他从业人员",
                            "all_l2": {
                                "8-00": "不便分类的其他从业人员"
                            }
                        }
                    }

                    # 动态渲染高可读性导航网（压缩版：L1完整+L2只显示代码）
                    sb = ["【🚨 中华人民共和国职业分类大典（2022修订版）宏观导航手册（压缩版）】"]
                    sb.append("（说明：L1为完整描述，L2只显示代码。如不清楚具体L2含义，请输出 REQ_L2 申请查阅详情！）")
                    for l1_code, info in macro_structure.items():
                        l1_name = info['name']
                        l2_codes = ", ".join(info['all_l2'].keys())
                        sb.append(f"L1-{l1_code}: {l1_name}")
                        sb.append(f"    官方定义：{info.get('definition', '')}")
                        sb.append(f"  L2: {l2_codes}")
                        sb.append("")  # 空行分隔
                    global_outline_text = "\n".join(sb)

                    # 拼装增量记忆上下文
                    ledger_text = "\n".join(exploration_ledger) if exploration_ledger else "无"

                    # P2: 收敛度诊断信号（增强版：三级响应）
                    conv_7d = hop_audit.get("convergence_7d", 0)
                    drift_cnt = hop_audit.get("drift_count", 0)
                    risk_lv = hop_audit.get("risk_level", "LOW")
                    diag_lines = [
                        "    【系统诊断：收敛度信号】",
                        f"    当前 Hop-{current_hop}，证据收敛度 = {conv_7d}/6",
                        f"    方向稳定性 = {'稳定' if drift_cnt == 0 else f'已漂移{drift_cnt}次'}",
                        f"    风险等级 = {risk_lv}",
                    ]
                    if conv_7d <= 2:
                        diag_lines.append("    🚨 收敛度极低！候选池与JD可能存在根本性不匹配。")
                        diag_lines.append("    🚨 强制建议：重新检查大类方向，强烈考虑换路或 GLOBAL_SEARCH！")
                        # 额外注入：推荐大类备选
                        if drift_cnt >= 1:
                            diag_lines.append(f"    🚨 已漂移{drift_cnt}次，说明此前方向选择存在问题，请回溯！")
                    elif conv_7d < 4:
                        diag_lines.append("    ⚠️ 收敛度偏低，建议重新审视大类选择是否正确。")
                        diag_lines.append("    若前几跳结论存在矛盾，请考虑回溯换路或 REQ_L2 切换大类。")
                    else:
                        diag_lines.append("    ✅ 收敛度正常，请继续。")
                    diag_block = "\n".join(diag_lines) + "\n"
                    memory_block = f"""【🧭 你的全局探索轨迹账本 (防循环死锁)】
    你之前已经探索过以下路径并得出了结论，请避免重复请求已被否定的代码大类：
    {ledger_text}

    【当前跳反思记录】
    {json.dumps(reasoning, ensure_ascii=False, indent=2)}

    {global_outline_text}

    🎯 【下一步行动核心指令】：
    当前候选池已因向量召回偏差被判定为「严重场景绑架/偏离真实产业」。请基于上方的【全局探索轨迹账本】避开已否定的路径，并在【全局宏观导航手册】中重新定位最贴切的二级大类。
    
    ⚠️ 【重要提醒】：宏观导航手册中 L2 只显示了代码（如 2-01, 2-02），没有显示完整名称。如果您不清楚某个 L2 代码的具体含义，**请立即输出动作 `REQ_L2`，系统将为您翻开该 L2 大类的完整定义和细分列表**！
    
    定位后，请立即输出动作 `REQ_L2`，并在 `result_code` 中填入该二级前缀（例如：action 填 "REQ_L2"，result_code 填 "2-02"）。系统将无损为您翻开该大类下所有的细分卡片！"""

                    current_user_prompt = f"{base_user_prompt}\n\n======================\n{memory_block}"
                    current_hop += 1
                    continue
                
                # ---------------------------------------------
                # 分支 4：请求调阅二级全景 (去噪骨架版，防 OOM)
                # ---------------------------------------------
                elif action == "REQ_L2":
                    print(f"  ⚠️ [Hop {current_hop}] 触发 L2 全景调阅: {result_code}")
                
                    if current_hop >= self.max_hops:
                        print(f"  💥 [Hop {current_hop}] 达到最大跳数限制，熔断退出。转人工复核。")
                        return {"status": "MELTDOWN", "code": "MANUAL_REVIEW", "confidence": 0, "reasoning": reasoning, "log": trajectory_log}
                    
                    # 🔧 修复：检测 1 类（管理层）死循环
                    # 若已有 3 跳在 1 类路径上徘徊且没有 FINALIZE，判定为"管理层归类困难"
                    # 强制降级处理：若 1 类所有路径都探索过，返回 REVIEW_SUGGESTED 而非死循环
                    requested_l2_str = str(result_code)
                    l1_class_attempts = sum(1 for e in exploration_ledger if e.startswith('[') and 'REQ_L2(1-' in e)
                    if requested_l2_str.startswith('1-') and l1_class_attempts >= 3:
                        print(f"  🔶 [Hop {current_hop}] 1类管理层路径已探索 {l1_class_attempts} 次，仍无法定谳。")
                        print(f"  🔶 强制终止循环，以最近一次 1-06 推理结果作为建议结果（REVIEW_SUGGESTED）。")
                        # 从探索账本中提取最近的 1 类代码作为兜底
                        last_1_class = "1-06-01"  # 默认兜底为企事业单位负责人
                        for entry in reversed(exploration_ledger):
                            import re as _re2
                            m = _re2.search(r'\((\d-\d{2}-\d{2}[^)]*)\)', entry)
                            if m and m.group(1).startswith('1-'):
                                last_1_class = m.group(1)
                                break
                        return {
                            "status": "REVIEW_SUGGESTED", 
                            "code": last_1_class, 
                            "confidence": 45,
                            "reasoning": {**reasoning, "_note": "1类管理层死循环，强制终止，建议人工复核"},
                            "log": trajectory_log
                        }
                
                    requested_l2_codes = re.findall(r'\d-\d{2}', str(result_code))
                    if not requested_l2_codes:
                        print("  ❌ 无法解析请求查阅的代码格式。")
                        current_hop += 1
                        continue

                    print(f"  🔍 正在无损拉取 {requested_l2_codes} 旗下全量名称与定义...")
                    l2_features = []
                    l3_codes = []  # 只存储 L3 代码
                    
                    for l2_code in requested_l2_codes:
                        for item in self.retriever.corpus:
                            if item['code'].startswith(l2_code):
                                # 🚨 分层加载：只显示 L3，L4 延迟到 REQ_L3_FULL
                                if re.match(r'^\d-\d{2}-\d{2}$', item['code']):
                                    clean_text = re.sub(r'主要工作任务.*', '', item['text'], flags=re.DOTALL).strip()
                                    l3_codes.append(f"[{item['code']}] {clean_text}")
                    
                    # 添加 L3 代码
                    l2_features.extend(l3_codes)
                    
                    # 添加延迟加载提示
                    if l3_codes:
                        l2_features.append("")
                        l2_features.append("【🚨 提示】：以上只显示三级代码。如需查看某个三级代码下的四级细节，请输出 REQ_L3_FULL 并填入该三级代码（例如：action=\"REQ_L3_FULL\", result_code=[\"2-02-10\"]）！")
                    
                    full_dict_context = "\n".join(l2_features) if l2_features else f"⚠️ 系统未找到大类 {requested_l2_codes}。"
                
                    # 🌟【核心修复】REQ_L2 二次向量召回：在该 L2 范围内做 Reranker 打分
                    # 避免 LLM 纯靠名称推理导致误判（瓶颈2的修复）
                    scoped_recall_context = ""
                    try:
                        scoped_recall_context = self.retriever.retrieve_scoped(
                            job_name, job_desc, requested_l2_codes
                        )
                        if scoped_recall_context:
                            print(f"  🎯 [二次召回] 已在该 L2 范围内完成 Reranker 交叉打分，注入 Top-K 精准候选")
                        else:
                            print(f"  ⚠️ [二次召回] 该 L2 范围内无匹配候选（向量分数全低于阈值）")
                    except Exception as e:
                        print(f"  ⚠️ [二次召回] 执行异常: {e}，降级为纯定义模式")
                
                    # 🌟 挂载上一步反思与多向选择权
                    ledger_text = "\n".join(exploration_ledger) if exploration_ledger else "无"
                
                    # 组装记忆模块：优先展示向量召回结果，再展示全量定义
                    # 🌟 scoped_recall_context 开头已包含置信度信号（警报或正常值）
                    if scoped_recall_context:

                        # P2: 收敛度诊断信号
                        conv_7d = hop_audit.get("convergence_7d", 0)
                        drift_cnt = hop_audit.get("drift_count", 0)
                        risk_lv = hop_audit.get("risk_level", "LOW")
                        diag_lines = [
                            "    【系统诊断：收敛度信号】",
                            f"    当前 Hop-{current_hop}，证据收敛度 = {conv_7d}/6",
                            f"    方向稳定性 = {'稳定' if drift_cnt == 0 else f'已漂移{drift_cnt}次'}",
                            f"    风险等级 = {risk_lv}",
                        ]
                        if conv_7d <= 2:
                            diag_lines.append("    🚨 收敛度极低！候选池与JD可能存在根本性不匹配。")
                            diag_lines.append("    🚨 强制建议：重新检查大类方向，强烈考虑换路或 GLOBAL_SEARCH！")
                            if drift_cnt >= 1:
                                diag_lines.append(f"    🚨 已漂移{drift_cnt}次，说明此前方向选择存在问题，请回溯！")
                        elif conv_7d < 4:
                            diag_lines.append("    ⚠️ 收敛度偏低，建议重新审视大类选择是否正确。")
                            diag_lines.append("    若前几跳结论存在矛盾，请考虑回溯换路或 REQ_L2 切换大类。")
                        else:
                            diag_lines.append("    ✅ 收敛度正常，请继续。")
                        diag_block = "\n".join(diag_lines) + "\n"
                        memory_block = f"""【🧭 你的全局探索轨迹账本 (防循环死锁)】
    你之前已经探索过以下路径并得出了结论，请避免重复请求已被否定的代码大类：
    {ledger_text}

    【当前跳反思记录】
    {json.dumps(reasoning, ensure_ascii=False, indent=2)}

    【🎯 系统响应：二次向量精准召回（Reranker 交叉注意力打分）】
    系统已在 {requested_l2_codes} 范围内用 Reranker 重新打分。
    ⚠️ 请先仔细阅读最上方的「置信度信号」，再查看候选列表：
    {scoped_recall_context}

    请优先基于以上候选做七维交叉验证。若以上候选全部不匹配，可再查阅下方全量定义列表。

    【系统响应：大类全景定义（全量备查）】
    以下是 {requested_l2_codes} 旗下所有的三级与四级细分：
    {full_dict_context}

    {diag_block}
                    🎯 【下一步行动指令 (宏观多向选择)】：
                    1. 🎯【首选】对上方「二次向量精准召回」中的候选做七维交叉验证，若匹配则直接 【FINALIZE】。
                    2. 🚨 若置信度警报触发（低分），必须在 Step 3 独立假说中讨论是否选错大类，如可疑就输出 action=\"REQ_L2\" 换路。
                    3. 若向量召回候选均不匹配但下方定义列表中有合适选项，可自行选择后 【FINALIZE】。
                    4. 若在当前大类的 2-3 个三级选项间犹豫，请输出动作 【REQ_L3_FULL】 调阅其微观任务清单。
                    5. 🚨 若确认当前大类错误，可基于上方的【探索账本】避开雷区，再次输出 【REQ_L2】 指定其他二级前缀。
                    6. 🚨 【终极逃逸】：若彻底迷失方向或发现与真实产业发生严重偏离，请果断输出动作 【GLOBAL_SEARCH】。

                    【⚡ 输出格式强制提醒（不看此条即为失职）】
                    必须严格按照 JSON_COT_TEMPLATE 格式输出，不得输出 squad_analysis、final_result 或其他任何格式！

                    正确格式（必须严格遵守）：
                    {{
                      "reasoning": {{
                        "step_0_fact_extraction": "...",
                        "step_1_macro_class": "...",
                        "step_2_hypothesis": "...",
                        "step_3_evidence_cross_match": {{...}},
                        "step_4_final_decision": "..."
                      }},
                      "action": "FINALIZE",
                      "result_code": "...",
                      "confidence_score": 85,
                      "is_rule_forced": false
                    }}

                    ❌ 错误格式（严禁输出）：
                    {{"squad_analysis": {{...}}, "final_result": {{...}}}}

                    现在开始输出第 {current_hop} 跳的推理结果："""
                    else:

                        # P2: 收敛度诊断信号
                        conv_7d = hop_audit.get("convergence_7d", 0)
                        drift_cnt = hop_audit.get("drift_count", 0)
                        risk_lv = hop_audit.get("risk_level", "LOW")
                        diag_lines = [
                            "    【系统诊断：收敛度信号】",
                            f"    当前 Hop-{current_hop}，证据收敛度 = {conv_7d}/6",
                            f"    方向稳定性 = {'稳定' if drift_cnt == 0 else f'已漂移{drift_cnt}次'}",
                            f"    风险等级 = {risk_lv}",
                        ]
                        if conv_7d <= 2:
                            diag_lines.append("    🚨 收敛度极低！候选池与JD可能存在根本性不匹配。")
                            diag_lines.append("    🚨 强制建议：重新检查大类方向，强烈考虑换路或 GLOBAL_SEARCH！")
                            if drift_cnt >= 1:
                                diag_lines.append(f"    🚨 已漂移{drift_cnt}次，说明此前方向选择存在问题，请回溯！")
                        elif conv_7d < 4:
                            diag_lines.append("    ⚠️ 收敛度偏低，建议重新审视大类选择是否正确。")
                            diag_lines.append("    若前几跳结论存在矛盾，请考虑回溯换路或 REQ_L2 切换大类。")
                        else:
                            diag_lines.append("    ✅ 收敛度正常，请继续。")
                        diag_block = "\n".join(diag_lines) + "\n"
                        memory_block = f"""【🧭 你的全局探索轨迹账本 (防循环死锁)】
    你之前已经探索过以下路径并得出了结论，请避免重复请求已被否定的代码大类：
    {ledger_text}

    【当前跳反思记录】
    {json.dumps(reasoning, ensure_ascii=False, indent=2)}

    【系统响应：大类全景查阅结果】
    以下是 {requested_l2_codes} 旗下所有的三级与四级细分：
    {full_dict_context}

    {diag_block}
    🎯 【下一步行动指令 (宏观多向选择)】：
    1. 若定位到目标或确认无任何对应细分，请直接 【FINALIZE】（可依法启用 99 兜底）。
    2. 若在当前大类的 2-3 个三级选项间犹豫，请输出动作 【REQ_L3_FULL】 调阅其微观任务清单。
    3. 🚨 若确认当前大类错误，可基于上方的【探索账本】避开雷区，再次输出 【REQ_L2】 指定其他二级前缀。
    4. 🚨 【终极逃逸】：若彻底迷失方向或发现与真实产业发生严重偏离，请果断输出动作 【GLOBAL_SEARCH】（result_code 亦填 "GLOBAL_SEARCH"），呼叫全量导航网重新寻路！

    【⚡ 输出格式强制提醒（不看此条即为失职）】
    必须严格按照 JSON_COT_TEMPLATE 格式输出，不得输出 squad_analysis、final_result 或其他任何格式！

    正确格式（必须严格遵守）：
    {{
      "reasoning": {{
        "step_0_fact_extraction": "...",
        "step_1_macro_class": "...",
        "step_2_hypothesis": "...",
        "step_3_evidence_cross_match": {{...}},
        "step_4_final_decision": "..."
      }},
      "action": "FINALIZE",
      "result_code": "...",
      "confidence_score": 85,
      "is_rule_forced": false
    }}

    ❌ 错误格式（严禁输出）：
    {{"squad_analysis": {{...}}, "final_result": {{...}}}}

    现在开始输出第 {current_hop} 跳的推理结果："""
                
                    # 重新拼装下一跳上下文：基石 Context + 记忆增量模块
                    current_user_prompt = f"{base_user_prompt}\n\n======================\n{memory_block}"
                    current_hop += 1
            
                # ---------------------------------------------
                # 分支 4：请求深挖三级详情 (全量底牌翻开)
                # ---------------------------------------------
                elif action == "REQ_L3_FULL":
                    print(f"  ⚠️ [Hop {current_hop}] 触发 L3 微观深挖: {result_code}")
                
                    if current_hop >= self.max_hops:
                        print(f"  💥 [Hop {current_hop}] 达到最大跳数限制，熔断退出。转人工复核。")
                        return {"status": "MELTDOWN", "code": "MANUAL_REVIEW", "confidence": 0, "reasoning": reasoning, "log": trajectory_log}
                
                    # 兼容列表或字符串提取
                    req_l3_list = result_code if isinstance(result_code, list) else re.findall(r'\d-\d{2}-\d{2}', str(result_code))
                
                    print(f"  🔍 正在翻开 {req_l3_list} 及其四级的全量底牌(包含主要工作任务)...")
                    l3_full_features = []
                    for l3_code in req_l3_list:
                        for item in self.retriever.corpus:
                            if item['code'].startswith(l3_code):
                                # 🚨 微观深潜：绝不截断！保留最原始包含几千字工作任务的全量文本
                                l3_full_features.append(f"[{item['code']}] {item['text']}")
                            
                    full_detail_context = "\n\n".join(l3_full_features) if l3_full_features else f"⚠️ 未找到代码 {req_l3_list} 的详情。"
                
                    # 🌟 挂载上一步反思与回退权
                    ledger_text = "\n".join(exploration_ledger) if exploration_ledger else "无"

                    # P2: 收敛度诊断信号（增强版：三级响应）
                    conv_7d = hop_audit.get("convergence_7d", 0)
                    drift_cnt = hop_audit.get("drift_count", 0)
                    risk_lv = hop_audit.get("risk_level", "LOW")
                    diag_lines = [
                        "    【系统诊断：收敛度信号】",
                        f"    当前 Hop-{current_hop}，证据收敛度 = {conv_7d}/6",
                        f"    方向稳定性 = {'稳定' if drift_cnt == 0 else f'已漂移{drift_cnt}次'}",
                        f"    风险等级 = {risk_lv}",
                    ]
                    if conv_7d <= 2:
                        diag_lines.append("    🚨 收敛度极低！候选池与JD可能存在根本性不匹配。")
                        diag_lines.append("    🚨 强制建议：重新检查大类方向，强烈考虑换路或 GLOBAL_SEARCH！")
                        # 额外注入：推荐大类备选
                        if drift_cnt >= 1:
                            diag_lines.append(f"    🚨 已漂移{drift_cnt}次，说明此前方向选择存在问题，请回溯！")
                    elif conv_7d < 4:
                        diag_lines.append("    ⚠️ 收敛度偏低，建议重新审视大类选择是否正确。")
                        diag_lines.append("    若前几跳结论存在矛盾，请考虑回溯换路或 REQ_L2 切换大类。")
                    else:
                        diag_lines.append("    ✅ 收敛度正常，请继续。")
                    diag_block = "\n".join(diag_lines) + "\n"
                    memory_block = f"""【🧭 你的全局探索轨迹账本 (防循环死锁)】
    你之前已经探索过以下路径并得出了结论，请避免重复请求已被否定的代码：
    {ledger_text}

    【当前跳反思记录】
    {json.dumps(reasoning, ensure_ascii=False, indent=2)}

    【系统响应：微观详情查阅结果】
    以下是你申请查阅的 {req_l3_list} 及其四级全量「主要工作任务」清单：
    {full_detail_context}

    {diag_block}
    🎯 【下一步行动指令 (微观终局抉择)】：
    1. 若细节动作契合且证据确凿，请结合上述微观任务，直接输出 【FINALIZE】 锁定对应代码。
    2. 🚨 【红线纠偏】：若发现微观细节不符，方向彻底错误：
       - 请查阅上方的【探索账本】排除死路，输出 【REQ_L2】 强行跃迁回宏观视角填入新的大类。
       - 若彻底迷失方向，请果断输出动作 【GLOBAL_SEARCH】 重置候选池，呼叫导航手册重新定位！"""
                
                    current_user_prompt = f"{base_user_prompt}\n\n======================\n{memory_block}"
                    current_hop += 1

                # ---------------------------------------------
                # 分支 5：精准对比裁决（COMPARE — P3 新增）
                # ---------------------------------------------
                elif action == "COMPARE":
                    compare_codes = result_code if isinstance(result_code, list) else []
                    # 过滤非法格式
                    compare_codes = [c for c in compare_codes if isinstance(c, str) and re.match(r'\d-\d{2}-\d{2}', c)]
                    if len(compare_codes) < 2:
                        print(f"  ⚠️ [COMPARE] result_code 格式无效或不足2个: {result_code}，降级为 REQ_L3_FULL")
                        # 降级处理
                        if compare_codes:
                            result_code = compare_codes
                        action = "REQ_L3_FULL"
                        continue  # 回循环顶部，走 REQ_L3_FULL 分支

                    print(f"  🔬 [Hop {current_hop}] COMPARE 精准对比: {compare_codes}")

                    if current_hop >= self.max_hops:
                        print(f"  💥 [Hop {current_hop}] 达到最大跳数限制，熔断退出。")
                        return {"status": "MELTDOWN", "code": "MANUAL_REVIEW", "confidence": 0, "reasoning": reasoning, "log": trajectory_log}

                    # 构建对比表
                    compare_table = self._build_compare_table(compare_codes, job_name, job_desc)

                    ledger_text = "\n".join(exploration_ledger) if exploration_ledger else "无"

                    memory_block = f"""【🧭 你的全局探索轨迹账本 (防循环死锁)】
    你之前已经探索过以下路径并得出了结论，请避免重复请求已被否定的代码：
    {ledger_text}

    【当前跳反思记录】
    {json.dumps(reasoning, ensure_ascii=False, indent=2)}

    🔬🔬🔬 【系统响应：精准对比裁决台】 🔬🔬🔬
    以下是你请求对比的 {len(compare_codes)} 个候选代码的并排诊断信息：

    {compare_table}

    🎯 【下一步行动指令（必须 FINALIZE）】：
    1. 基于上方的并排对比信息，结合 step_0 的劳动事实做最终裁决
    2. 你必须在本跳输出 action="FINALIZE"（COMPARE 是最后一站，不再回溯）
    3. 若对比后仍无法裁决：选最接近的候选 FINALIZE，confidence_score 填低分（<60）
    4. 🚨 若确认这 {len(compare_codes)} 个全部错误 → FINALIZE 时使用你认为正确的代码

    【⚡ 输出格式强制提醒】
    必须严格按照 JSON_COT_TEMPLATE 格式输出 FINALIZE！"""
                    current_user_prompt = f"{base_user_prompt}\n\n======================\n{memory_block}"
                    current_hop += 1

                else:
                    print(f"  ❓ 未知 Action: {action}")
                    current_hop += 1

            return {
                "status": "MELTDOWN", 
                "code": "MANUAL_REVIEW", 
                "confidence": 0,
                "reasoning": {"error": "已耗尽最大跳数，或模型连续输出未知指令"}, 
                "log": trajectory_log
            }
        except Exception as e:
            tb_text = traceback.format_exc()
            print(f"  💥 [FATAL] run() 异常: {e}")
            print(f"{tb_text}")
            trajectory_log.append(f"=== SYSTEM_CRASH ===\n{tb_text}")
            return {
                "status": "SYSTEM_CRASH",
                "code": "FATAL",
                "confidence": 0,
                "reasoning": {"error": str(e), "traceback": tb_text},
                "log": trajectory_log
            }
