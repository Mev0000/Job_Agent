import json
import re

class JobAgentStateMachine:
    def __init__(self, llm_client, retriever, config, rules_prompt, json_template):
        """
        初始化 Agent 状态机
        :param llm_client: Gemma4Client 实例
        :param retriever: 双轨融合引擎实例 (内部自带 graph_rag)
        ...
        """
        self.llm = llm_client
        self.retriever = retriever          
        self.graph_rag = retriever.graph    
        self.max_hops = config.get("max_hops", 3)
        
        self.system_prompt = f"{rules_prompt}\n\n{json_template}"
        self.global_dict_tree = getattr(self.graph_rag, 'global_dict_tree', {})

    def _extract_json_from_text(self, text):
        """尝试从模型输出的文本中提取合法的 JSON，带 DEBUG 监控"""
        # 👇 核心在这里：不管三七二十一，先打印原始文本
        print(f"\n[🚨 DEBUG 监控] 大模型的原始输出如下：\n{text}\n{'='*40}")
        
        try:
            # 找到最外层的 {}
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                return None
        except Exception as e:
            print(f"⚠️ 模型输出解析 JSON 失败: {e}")
            return None

    def run(self, job_name, job_desc, initial_candidates_context):
        """
        运行状态机主循环
        :param job_name: 岗位名称
        :param job_desc: 岗位描述
        :param initial_candidates_context: 第一跳(Hop-1)通过 BGE-M3 检索回来的候选文本块
        """
        current_hop = 1
        
        # 初始 User Prompt
        base_user_prompt = f"""【目标岗位】\n名称：{job_name}\n职责：{job_desc}\n\n【初筛参考库 (Hop-1)】\n{initial_candidates_context}"""
        current_user_prompt = base_user_prompt
        
        # 记录每一跳的思考过程，用于最终输出日志
        trajectory_log = []

        while current_hop <= self.max_hops:
            print(f"  ➡️ 开始第 {current_hop} 跳推理...")
            
            # 1. 调用模型
            response_text = self.llm.generate(self.system_prompt, current_user_prompt)
            
            if not response_text:
                return {"status": "ERROR", "message": "模型无响应", "log": trajectory_log}
                
            trajectory_log.append(f"=== Hop {current_hop} 模型原始输出 ===\n{response_text}\n")
            
            # 2. 解析 JSON
            result_json = self._extract_json_from_text(response_text)
            
            if not result_json or 'action' not in result_json:
                print("  ❌ 未能解析出标准 JSON 或缺少 action 字段，尝试重试。")
                current_hop += 1
                continue
                
            action = result_json.get('action')
            result_code = result_json.get('result_code', 'UNKNOWN')
            
            # 3. 状态路由判断
            if action == "FINALIZE":
                print(f"  ✅ [Hop {current_hop}] 成功定谳，代码: {result_code}")
                return {
                    "status": "SUCCESS", 
                    "code": result_code, 
                    "reasoning": result_json.get("reasoning"),
                    "log": trajectory_log
                }
                
            elif action == "8-00-00":
                print(f"  🗑️ [Hop {current_hop}] 判定为空壳/垃圾岗位。")
                return {
                    "status": "GARBAGE", 
                    "code": "8-00-00", 
                    "reasoning": result_json.get("reasoning"),
                    "log": trajectory_log
                }
                
            elif action == "REQ_C":
                print(f"  ⚠️ [Hop {current_hop}] 模型触发拦截，申请查阅: {result_code}")
                
                if current_hop >= self.max_hops:
                    print(f"  💥 [Hop {current_hop}] 达到最大跳数限制，熔断退出。转人工复核。")
                    return {
                        "status": "MELTDOWN", 
                        "code": "MANUAL_REVIEW", 
                        "reasoning": result_json.get("reasoning"),
                        "log": trajectory_log
                    }
                
                # ==== 核心策略 B：带着记忆准备下一跳 ====
                # 提取模型请求的二级代码 (假设 result_code 格式类似 "2-02" 或者是 "[2-02, 3-01]")
                requested_l2_codes = re.findall(r'\d-\d{2}', str(result_code))
                
                if not requested_l2_codes:
                    print("  ❌ 无法解析请求查阅的代码格式。")
                    return {"status": "ERROR", "message": "请求查阅代码格式错误", "log": trajectory_log}

                print(f"  🔍 正在为您拉取全量字典特征: {requested_l2_codes} ...")
                
                # (你需要确保 global_dict_tree 有这个提取全量三四级特征的方法，这类似于你原脚本里的提取逻辑)
                # full_dict_context = self.graph_rag.get_full_l2_features(requested_l2_codes) 
                
                # 为了演示，此处放个占位符
                full_dict_context = f"[这里是系统拉取的 {requested_l2_codes} 的全量大典细分特征...]" 
                
                # 重组下一跳的 User Prompt (策略 B：带着错题本)
                memory_block = f"""【上一跳(Hop-{current_hop})反思记录】\n系统在初筛库中发生了场景绑架或证据缺失。你的上一步推理结论是：\n{json.dumps(result_json.get('reasoning'), ensure_ascii=False, indent=2)}\n\n【系统响应：全量字典调阅】\n已为您调出所需二级分类的全部特征。请重新执行 5 步审判，务必输出一个最终的 7 位三级代码：\n\n{full_dict_context}"""
                
                current_user_prompt = f"【目标岗位】\n名称：{job_name}\n职责：{job_desc}\n\n{memory_block}"
                
                current_hop += 1
            
            else:
                print(f"  ❓ 未知 Action: {action}")
                current_hop += 1

        return {"status": "ERROR", "message": "循环异常退出", "log": trajectory_log}