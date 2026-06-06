# core/state_machine.py

import json
import re
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
            "threshold_review": 75,  # 低于此分数标记为REVIEW_SUGGESTED
            "score_full_match": 30,   # 三维全同构加分
            "score_rule_hit": 20,       # 口诀强制命中加分
            "penalty_99": -25,          # 使用99兜底减分
            "penalty_p1_mismatch": -20,  # P1不一致但仍选择减分
            "penalty_global_search": -10   # 经过GLOBAL_SEARCH重定向减分
        })

    def _extract_json_from_text(self, text):
        """增强版 JSON 提取器，防 Markdown 干扰与非法转义"""
        # 🌟 修复 1：取消截断，打印全量日志以便排查！
        print(f"\n[🚨 DEBUG 监控] 模型输出 (全量展示)：\n{text}\n{'='*40}")
        
        try:
            # 🌟 修复 2：预处理非法转义符。大模型有时会输出未转义的反斜杠(如 \s, \d 或者单纯的 \ )
            # 这会导致 json.loads 报 Invalid \escape 错误。我们将单反斜杠替换为双反斜杠进行安全逃逸。
            clean_text = text.replace('\\', '\\\\')
            
            # 策略 1：优先匹配被 ```json ... ``` 包裹的内容
            match = re.search(r'```json\s*(.*?)\s*```', clean_text, re.DOTALL)
            if match:
                # strict=False 允许 JSON 中包含某些非法的控制字符（如真实的换行符）
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
        
        评分规则（可配置，见config.yaml的agent.confidence）：
        基础分: 50
        +30分: 三维全同构（交付物+核心动作+作用对象均"一致"/"匹配"）
        +20分: 口诀强制命中（is_rule_forced=True，由检索层传入）[P1-A]
        -25分: 使用99兜底代码
        -10分: 经过GLOBAL_SEARCH重定向（通过trajectory_log轨迹匹配）[P1-B]
        
        返回：0-100的整数置信度评分
        """
        score = 50  # 基础分
        
        # 获取配置
        cfg = self.confidence_config
        score_full = cfg.get("score_full_match", 30)
        score_rule = cfg.get("score_rule_hit", 20)
        penalty_99 = cfg.get("penalty_99", -25)
        penalty_gs = cfg.get("penalty_global_search", -10)
        
        # 检查是否使用99兜底
        result_code = result_json.get("result_code", "")
        if result_code.endswith("99"):
            score += penalty_99
        
        # P1-B: 用trajectory_log轨迹字符串匹配，替代 hop > 3 的粗糙启发式判断
        # 精确检测是否曾经触发过 GLOBAL_SEARCH 动作（说明初始方向错误，重新定向）
        if trajectory_log:
            if any("GLOBAL_SEARCH" in step for step in trajectory_log):
                score += penalty_gs
        elif current_hop > 3:
            # 兼容旧调用（无trajectory_log传入时的降级方案）
            score += penalty_gs
            
        # P1-A: 若检索层传入了 is_rule_forced=True（图谱口诀强制命中），加分
        # is_rule_forced 由 retriever.retrieve() 在候选字典的元数据中传递
        if is_rule_forced:
            score += score_rule
            
        # 检查step_4的结构化比对结果（如果存在），计算三维全同构加分
        step4 = reasoning.get("step_4_evidence_cross_match", {})
        if isinstance(step4, dict):
            check1 = str(step4.get("check_1_deliverables", ""))
            check2 = str(step4.get("check_2_core_actions", ""))
            check3 = str(step4.get("check_3_objects", ""))
            
            # 三维全同构判断：三项均包含"一致"或"匹配"关键词
            positive_keywords = {"一致", "匹配", "符合", "吻合", "契合"}
            def _is_positive(text: str) -> bool:
                return any(kw in text for kw in positive_keywords)
            
            if _is_positive(check1) and _is_positive(check2) and _is_positive(check3):
                score += score_full
                
        # 确保分数在0-100范围内
        score = max(0, min(100, score))
        return score
            
    def run(self, job_name, job_desc, initial_candidates_context):
        """
        运行状态机主循环
        """
        current_hop = 1
        
        # 将最原始的 JD 和初筛库封装为基石 Prompt，防止后续跳跃中失忆
        base_user_prompt = f"【目标岗位】\n名称：{job_name}\n职责：{job_desc}\n\n【初筛参考库 (Hop-1)】\n{initial_candidates_context}"
        current_user_prompt = base_user_prompt
        
        trajectory_log = []
        exploration_ledger = []

        while current_hop <= self.max_hops:
            print(f"  ➡️ 开始第 {current_hop} 跳推理...")
            
            # 1. 调用大模型
            response_text = self.llm.generate(self.system_prompt, current_user_prompt)
            
            if not response_text:
                return {"status": "ERROR", "message": "模型无响应", "log": trajectory_log}
                
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
            
            step_summary = reasoning.get("step_5_isomorphic_finalize", "未完成定谳")
            exploration_ledger.append(f"-> [第 {current_hop} 跳尝试]: 动作 {action} ({result_code}) | 结论: {step_summary}")
            
            # ---------------------------------------------
            # 分支 1：成功定谳
            # ---------------------------------------------
            if action == "FINALIZE":
                print(f"  ✅ [Hop {current_hop}] 成功定谳，代码: {result_code}")
                
                # 计算置信度评分（客观规则，非LLM自评）
                # P1-A: 传入 is_rule_forced；P1-B: 传入 trajectory_log 用于精确轨迹匹配
                confidence = self._calculate_confidence(
                    result_json, reasoning, current_hop,
                    trajectory_log=trajectory_log,
                    is_rule_forced=is_rule_forced
                )
                
                # 根据置信度决定状态
                threshold = self.confidence_config.get("threshold_review", 75)
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
                        "all_l2": {
                            "3-01": "行政办事及辅助人员",
                            "3-02": "安全和消防及辅助人员",
                            "3-03": "法律事务及辅助人员",
                            "3-99": "其他办事人员和有关人员"
                        }
                    },
                    "4": {
                        "name": "社会生产服务和生活服务人员",
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

                # 动态渲染高可读性导航网
                sb = ["【🚨 中华人民共和国职业分类大典（2022修订版）全量宏观导航手册】"]
                for l1_code, info in macro_structure.items():
                    sb.append(f"第 {l1_code} 大类：{info['name']}")
                    for l2_code, l2_name in info['all_l2'].items():
                        sb.append(f"  ├── [{l2_code}] {l2_name}")
                global_outline_text = "\n".join(sb)

                # 拼装增量记忆上下文
                ledger_text = "\n".join(exploration_ledger) if exploration_ledger else "无"
                memory_block = f"""【🧭 你的全局探索轨迹账本 (防循环死锁)】
你之前已经探索过以下路径并得出了结论，请避免重复请求已被否定的代码大类：
{ledger_text}

【当前跳反思记录】
{json.dumps(reasoning, ensure_ascii=False, indent=2)}

{global_outline_text}

🎯 【下一步行动核心指令】：
当前候选池已因向量召回偏差被判定为「严重场景绑架/偏离真实产业」。请基于上方的【全局探索轨迹账本】避开已否定的路径，并在【全局宏观导航手册】中重新定位最贴切的二级大类。
定位后，请立即输出动作 `REQ_L2`，并在 `result_code` 中填入该二级前缀（例如：action 填 "REQ_L2"，result_code 填 "4-02"）。系统将无损为您翻开该大类下所有的细分卡片！"""

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
                
                requested_l2_codes = re.findall(r'\d-\d{2}', str(result_code))
                if not requested_l2_codes:
                    print("  ❌ 无法解析请求查阅的代码格式。")
                    current_hop += 1
                    continue

                print(f"  🔍 正在无损拉取 {requested_l2_codes} 旗下全量名称与定义...")
                l2_features = []
                for l2_code in requested_l2_codes:
                    for item in self.retriever.corpus:
                        # 只要前缀匹配，三级和四级一网打尽
                        if item['code'].startswith(l2_code):
                            # 🚨 动态降噪：强制抹除主要工作任务，只保留名称和核心定义
                            clean_text = re.sub(r'主要工作任务.*', '', item['text'], flags=re.DOTALL).strip()
                            l2_features.append(f"[{item['code']}] {clean_text}")
                
                full_dict_context = "\n".join(l2_features) if l2_features else f"⚠️ 系统未找到大类 {requested_l2_codes}。"
                
                # 🌟 挂载上一步反思与多向选择权
                ledger_text = "\n".join(exploration_ledger) if exploration_ledger else "无"
                memory_block = f"""【🧭 你的全局探索轨迹账本 (防循环死锁)】
你之前已经探索过以下路径并得出了结论，请避免重复请求已被否定的代码大类：
{ledger_text}

【当前跳反思记录】
{json.dumps(reasoning, ensure_ascii=False, indent=2)}

【系统响应：大类全景查阅结果】
以下是 {requested_l2_codes} 旗下所有的三级与四级细分：
{full_dict_context}

🎯 【下一步行动指令 (宏观多向选择)】：
1. 若定位到目标或确认无任何对应细分，请直接 【FINALIZE】（可依法启用 99 兜底）。
2. 若在当前大类的 2-3 个三级选项间犹豫，请输出动作 【REQ_L3_FULL】 调阅其微观任务清单。
3. 🚨 若确认当前大类错误，可基于上方的【探索账本】避开雷区，再次输出 【REQ_L2】 指定其他二级前缀。
4. 🚨 【终极逃逸】：若彻底迷失方向或发现与真实产业发生严重偏离，请果断输出动作 【GLOBAL_SEARCH】（result_code 亦填 "GLOBAL_SEARCH"），呼叫全量导航网重新寻路！"""
                
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
                memory_block = f"""【🧭 你的全局探索轨迹账本 (防循环死锁)】
你之前已经探索过以下路径并得出了结论，请避免重复请求已被否定的代码：
{ledger_text}

【当前跳反思记录】
{json.dumps(reasoning, ensure_ascii=False, indent=2)}

【系统响应：微观详情查阅结果】
以下是你申请查阅的 {req_l3_list} 及其四级全量「主要工作任务」清单：
{full_detail_context}

🎯 【下一步行动指令 (微观终局抉择)】：
1. 若细节动作契合且证据确凿，请结合上述微观任务，直接输出 【FINALIZE】 锁定对应代码。
2. 🚨 【红线纠偏】：若发现微观细节不符，方向彻底错误：
   - 请查阅上方的【探索账本】排除死路，输出 【REQ_L2】 强行跃迁回宏观视角填入新的大类。
   - 若彻底迷失方向，请果断输出动作 【GLOBAL_SEARCH】 重置候选池，呼叫导航手册重新定位！"""
                
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