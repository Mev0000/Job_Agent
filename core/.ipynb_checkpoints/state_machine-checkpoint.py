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
            # 分支 3： 请求全局大纲 
            # ---------------------------------------------
            elif action == "GLOBAL_SEARCH":
                print(f"  🚨 [Hop {current_hop}] 触发【上帝视角】：向量召回彻底失效，动态下发 2022 修订版全量一二级导航网...")
                
                # 完美对齐 2022 修订版大典官方最精准的一二级大类标准名称网
                macro_structure = {
                    "1": {
                        "name": "国家机关、党群组织、企业、事业单位负责人",
                        "all_l2": {
                            "1-01": "党的机关负责人", "1-02": "国家权力机关负责人", "1-03": "行政机关负责人",
                            "1-04": "政协组织负责人", "1-05": "审判和检察机关负责人", "1-06": "民主党派和工商联负责人",
                            "1-07": "人民团体和群众团体负责人", "1-08": "社会组织、中介机构和基金会负责人", "1-09": "企业、事业单位负责人"
                        }
                    },
                    "2": {
                        "name": "专业技术人员",
                        "all_l2": {
                            "2-01": "科学研究人员", "2-02": "工程技术人员", "2-03": "农业技术人员",
                            "2-04": "飞机和船舶技术人员", "2-05": "卫生专业技术人员", "2-06": "经济和金融专业人员",
                            "2-07": "法律、监察、社会工作专业人员", "2-08": "教学人员", "2-09": "文学艺术、体育和娱乐专业人员",
                            "2-10": "新闻出版、文化专业人员"
                        }
                    },
                    "3": {
                        "name": "办事人员和有关人员",
                        "all_l2": {
                            "3-01": "行政办公人员", "3-02": "安全保卫和消防办事人员", "3-03": "邮政和通信办事人员", "3-09": "其他办事人员"
                        }
                    },
                    "4": {
                        "name": "社会生产服务和生活服务人员",
                        "all_l2": {
                            "4-01": "批发零售服务人员", "4-02": "交通运输、仓储和邮政业服务人员", "4-03": "住宿和餐饮服务人员",
                            "4-04": "信息通信网络运行员和服务人员", "4-05": "金融服务人员", "4-06": "旅游服务人员",
                            "4-07": "社会生产服务人员", "4-08": "伴随服务人员", "4-09": "水利、环境和公共设施管理服务人员",
                            "4-10": "居民服务人员", "4-11": "电力、燃气及水供应服务人员", "4-12": "居民生活服务人员",
                            "4-13": "文化、体育和娱乐服务人员", "4-14": "健康、康复和医疗辅助服务人员"
                        }
                    },
                    "5": {
                        "name": "农、林、牧、渔业生产及辅助人员",
                        "all_l2": {
                            "5-01": "农业生产人员", "5-02": "林业生产人员", "5-03": "畜牧业生产人员",
                            "5-04": "渔业生产人员", "5-05": "农林牧渔业辅助人员", "5-09": "其他农林牧渔业生产及辅助人员"
                        }
                    },
                    "6": {
                        "name": "生产制造及有关人员",
                        "all_l2": {
                            "6-01": "勘探和矿物开采人员", "6-02": "金属冶炼和压延加工人员", "6-03": "化工产品生产人员",
                            "6-04": "机械制造加工人员", "6-05": "机电产品装配人员", "6-06": "电子信息及通信设备制造人员",
                            "6-07": "电力、热力、气体、水生产和输配人员", "6-08": "纺织、服装、皮革、毛皮制品加工制作人员",
                            "6-09": "缝纫、编织、工艺品 and 木制品制作人员", "6-10": "食品、饮料加工制作人员", "6-11": "烟草及其制品加工制作人员",
                            "6-12": "药品生产人员", "6-13": "木材加工、家具制造人员", "6-14": "造纸及纸制品加工制作人员",
                            "6-15": "建筑材料制造人员", "6-16": "硅酸盐制品制造人员", "6-17": "石油加工、炼焦、煤化工产品生产人员",
                            "6-18": "黑色金属冶炼及压延加工人员", "6-19": "有色金属冶炼及压延加工人员", "6-20": "金属制品制造人员",
                            "6-21": "通用设备制造人员", "6-22": "专用设备制造人员", "6-23": "交通运输设备制造人员",
                            "6-24": "电气机械及器材制造人员", "6-25": "计算机、通信和其他电子设备制造人员", "6-26": "仪器仪表制造人员",
                            "6-27": "废弃资源综合利用人员", "6-28": "电力、热力、燃气及水生产和供应人员", "6-29": "建筑施工人员",
                            "6-30": "运输设备操作人员及有关人员", "6-31": "检验、检测、计量、质量投放及有关人员"
                        }
                    },
                    "8": {
                        "name": "不便分类的其他劳动者",
                        "all_l2": {"8-00": "不便分类的其他劳动者"}
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
                memory_block = f"""【你的上一步反思记录 (继承保留)】
{json.dumps(reasoning, ensure_ascii=False, indent=2)}

{global_outline_text}

🎯 【下一步行动核心指令】：
当前候选池已因向量召回偏差被判定为「严重场景绑架/偏离真实产业」。请在上方的【全局宏观导航手册】中，重新定位最贴切的二级大类。
定位后，请立即输出动作 `REQ_L2`，并在 `result_code` 中填入该二级前缀（例如：你重新评估认为该岗位属于交通海运物流，则 action 填 "REQ_L2"，result_code 填 "4-02"）。系统将无损为您翻开该大类下所有的细分卡片！"""

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
以下是 {requested_l2_codes} 旗下所有的三级与四级细分：
{full_dict_context}

🎯 【下一步行动指令 (宏观多向选择)】：
1. 若定位到目标或确认无任何对应细分，请直接 【FINALIZE】（可依法启用 99 兜底）。
2. 若在当前大类的 2-3 个三级选项间犹豫，请输出动作 【REQ_L3_FULL】 调阅其微观任务清单。
3. 若确认当前二级职业大类错误但明确知晓其他大类，可再次输出 【REQ_L2】 并指定其他二级前缀（如 2-02）。
4. 🚨 【终极逃逸】：若翻阅当前二级职业大类全景后，发现当前方向与该岗位的真实产业/能力发生严重「场景绑架/彻底偏离」，请果断输出动作 【GLOBAL_SEARCH】（result_code 亦填 "GLOBAL_SEARCH"），系统将为您下发全量一二级大典骨架大纲进行重新寻路！"""
                
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
以下是你申请查阅的 {req_l3_list} 及其四级全量「主要工作任务」清单：
{full_detail_context}

🎯 【下一步行动指令 (微观终局抉择)】：
1. 若细节动作契合且证据确凿，请结合上述微观任务，直接输出 【FINALIZE】 锁定对应代码。
2. 🚨 【红线纠偏】：若翻开最底层的底牌（主要工作任务）后，发现被微观细节证伪，证明当前大类方向彻底错误或遭遇严重误导：
   - 若知晓正确二级职业大类，可输出 【REQ_L2】 强行跃迁回宏观视角并填入新的大类。
   - 若彻底迷失方向，请果断输出动作 【GLOBAL_SEARCH】 重置候选池，呼叫全量宏观导航手册重新定位！"""
                
                current_user_prompt = f"{base_user_prompt}\n\n======================\n{memory_block}"
                current_hop += 1

            else:
                print(f"  ❓ 未知 Action: {action}")
                current_hop += 1

        return {"status": "ERROR", "message": "循环异常退出", "log": trajectory_log}