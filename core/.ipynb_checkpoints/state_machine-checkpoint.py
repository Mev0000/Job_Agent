# core/state_machine.py

import json
import re

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
        # 为了支持充裕的横向跳跃与深潜，设定跳数上限为 4
        self.max_hops = config.get("max_hops", 4) 
        
        self.system_prompt = f"{rules_prompt}\n\n{json_template}"
        self.global_dict_tree = getattr(self.graph_rag, 'global_dict_tree', {})

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

    def run(self, job_name, job_desc, initial_candidates_context):
        """
        运行状态机主循环
        """
        current_hop = 1
        
        # 将最原始的 JD 和初筛库封装为基石 Prompt，防止后续跳跃中失忆
        base_user_prompt = f"【目标岗位】\n名称：{job_name}\n职责：{job_desc}\n\n【初筛参考库 (Hop-1)】\n{initial_candidates_context}"
        current_user_prompt = base_user_prompt
        
        trajectory_log = []

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
            
            # ---------------------------------------------
            # 分支 1：成功定谳
            # ---------------------------------------------
            if action == "FINALIZE":
                print(f"  ✅ [Hop {current_hop}] 成功定谳，代码: {result_code}")
                return {
                    "status": "SUCCESS", 
                    "code": result_code, 
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
                    "reasoning": reasoning,
                    "log": trajectory_log
                }
                
            # ---------------------------------------------
            # 分支 3：请求调阅二级全景 (去噪骨架版，防 OOM)
            # ---------------------------------------------
            elif action == "REQ_L2":
                print(f"  ⚠️ [Hop {current_hop}] 触发 L2 全景调阅: {result_code}")
                
                if current_hop >= self.max_hops:
                    print(f"  💥 [Hop {current_hop}] 达到最大跳数限制，熔断退出。转人工复核。")
                    return {"status": "MELTDOWN", "code": "MANUAL_REVIEW", "reasoning": reasoning, "log": trajectory_log}
                
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
                memory_block = f"""【你的上一步反思记录 (极其重要)】
你通过上一轮的推理得出了以下结论，请在接下来的判断中严格继承该推理：
{json.dumps(reasoning, ensure_ascii=False, indent=2)}

【系统响应：大类全景查阅结果】
以下是 {requested_l2_codes} 旗下所有的三级与四级细分 (已过滤繁杂任务，仅保留名称与定义)：
{full_dict_context}

🎯 【下一步行动指令 (多向选择)】：
1. 若找到目标或确认无任何对应细分，请直接 【FINALIZE】 (包括依法启用 99 兜底)。
2. 若在当前大类的 2-3 个选项间犹豫，请输出 【REQ_L3_FULL】 深挖细节。
3. 🚨 若发现当前大类方向完全错误，你可以再次输出 【REQ_L2】，去查阅其他二级大类！"""
                
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
                    return {"status": "MELTDOWN", "code": "MANUAL_REVIEW", "reasoning": reasoning, "log": trajectory_log}
                
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
                memory_block = f"""【你的上一步反思记录 (极其重要)】
你通过上一轮的推理得出了以下结论，请在接下来的判断中严格继承该推理：
{json.dumps(reasoning, ensure_ascii=False, indent=2)}

【系统响应：微观详情查阅结果】
以下是你申请查阅的 {req_l3_list} 及其四级全量「主要工作任务」：
{full_detail_context}

🎯 【下一步行动指令 (终局抉择)】：
1. 若证据确凿，请结合上述微观动作细节，直接 【FINALIZE】 定谳。
2. 🚨 若翻开底牌后发现依然不对，你可以输出 【REQ_L2】 退回到宏观视角，去查阅其他二级大类！"""
                
                current_user_prompt = f"{base_user_prompt}\n\n======================\n{memory_block}"
                current_hop += 1

            else:
                print(f"  ❓ 未知 Action: {action}")
                current_hop += 1

        return {"status": "ERROR", "message": "循环异常退出", "log": trajectory_log}