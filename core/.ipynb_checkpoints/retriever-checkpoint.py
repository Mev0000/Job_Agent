# core/retriever.py

import torch
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

# 引入重构后的图谱类
from core.graph_builder import DictGraphRAG

class AdvancedRetriever:
    def __init__(self, config: Dict[str, Any], occupation_corpus: List[Dict]):
        """
        初始化双轨安检式 RAG 检索器 (Dual-Track Auditing RAG)
        :param config: 全局配置字典
        :param occupation_corpus: 职业大典的语料列表，格式: [{"code": "4-01-02", "text": "营销员: 从事市场推销...", "name": "营销员"}]
        """
        self.config = config
        
        # ---------------- 1. 初始化 BGE-M3 向量引擎 ----------------
        model_path = config.get("retriever", {}).get("model_path", "BAAI/bge-m3")
        self.top_k_recall = config.get("retriever", {}).get("top_k_recall", 15) # 送入图谱安检的初始池大小
        
        print(f"🔄 [轨道 1] 正在加载 BGE-M3 向量模型: {model_path} ...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=device)
        
        self.corpus = occupation_corpus
        self.corpus_texts = [item['text'] for item in self.corpus]
        
        # 建立快速查找字典，供图谱强制拉人时使用
        self.corpus_dict = {item['code']: item for item in self.corpus}
        
        print(f"🔄 正在为 {len(self.corpus_texts)} 条职业大典语料建立向量索引...")
        self.corpus_embeddings = self.model.encode(self.corpus_texts, normalize_embeddings=True, show_progress_bar=True)
        
        # ---------------- 2. 初始化 GraphRAG 专家安检引擎 ----------------
        print(f"🔄 [轨道 2] 正在加载 GraphRAG 专家安检引擎...")
        graph_dir = config.get("graph", {}).get("data_dir", "data/graph_tables/")
        self.graph = DictGraphRAG(data_dir=graph_dir)
        
        # 尝试极速加载缓存，如果失败则全量构建
        cache_path = config.get("graph", {}).get("cache_path", "data/cache/graph_rag.pkl")
        try:
            self.graph.load_from_disk(cache_path)
        except:
            print("⚠️ 未找到图谱缓存，触发全量动态构建...")
            self.graph.build_graph()
            self.graph.save_to_disk(cache_path)

        print("✅ 双轨融合检索引擎上线完毕！")

    def retrieve(self, job_name: str, job_desc: str) -> str:
        """
        核心调度枢纽：执行 5 阶段双轨融合召回
        """
        # 阶段 1：BGE-M3 广撒网兜底 (向量路)
        vector_candidates = self._get_vector_candidates(job_name, job_desc, top_k=self.top_k_recall)
        
        # 阶段 2：GraphRAG 深度安检与强制绑定 (图谱路)
        audited_pool = self._audit_and_bind(vector_candidates)
        
        # 阶段 3：宁缺毋滥的断崖式阶梯截断
        final_candidates = self._cliff_cut(audited_pool)
        
        # 阶段 4：决战圈虫洞交叉排查 (Wormhole 互斥扫描)
        self._apply_wormhole_warnings(final_candidates)
        
        # 阶段 5：组装高密度信息胶囊 (喂给 LLM)
        final_prompt_context = self._format_prompt_context(final_candidates)
        
        return final_prompt_context

    def _get_vector_candidates(self, job_name: str, job_desc: str, top_k: int) -> List[Dict]:
        """【阶段 1】纯语义向量检索"""
        query_text = f"岗位名称：{job_name}。职责描述：{job_desc}"
        query_embedding = self.model.encode([query_text], normalize_embeddings=True)[0]
        
        similarities = np.dot(self.corpus_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        candidates = []
        for idx in top_indices:
            candidates.append({
                "code": self.corpus[idx]['code'],
                "name": self.corpus[idx].get('name', '未知职务'),
                "text": self.corpus[idx]['text'],
                "score": float(similarities[idx])
            })
        return candidates

    def _audit_and_bind(self, vector_candidates: List[Dict]) -> Dict[str, Dict]:
        """【阶段 2】图谱特征注入与易混淆强制绑定"""
        audited_pool = {}
        
        for cand in vector_candidates:
            code = cand['code']
            if code in audited_pool:
                continue
                
            features = self.graph.get_job_features(code)
            if not features:
                continue 
                
            audited_pool[code] = {
                "source": "Vector_Search",
                "score": cand['score'],
                "name": cand['name'],
                "text": cand['text'],
                "features": features,
                "confusion_warnings": set(),
                "wormhole_warnings": set() # 预留红线位
            }
            
            # API 2: 对决雷达 (强制绑定杀招)
            confused_codes = self.graph.get_confused_codes(code)
            for c_code in confused_codes:
                rule_text = self.graph.get_confusion_text(code, c_code)
                
                # 若混淆对手不在池内，强行拉入
                if c_code not in audited_pool:
                    c_features = self.graph.get_job_features(c_code)
                    c_corpus_data = self.corpus_dict.get(c_code)
                    
                    if c_features and c_corpus_data:
                        audited_pool[c_code] = {
                            "source": "GraphRAG_Force_Bind", 
                            "score": cand['score'] - 0.001, # 紧贴触发者，保证排序时不分离
                            "name": c_corpus_data.get('name', '未知职务'),
                            "text": c_corpus_data['text'],
                            "features": c_features,
                            "confusion_warnings": set(),
                            "wormhole_warnings": set()
                        }
                
                # 双向记录防坑口诀
                audited_pool[code]["confusion_warnings"].add(f"与 [{c_code}] 易混淆：{rule_text}")
                if c_code in audited_pool:
                    audited_pool[c_code]["confusion_warnings"].add(f"与 [{code}] 易混淆：{rule_text}")

        return audited_pool

    def _cliff_cut(self, pool: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        """【阶段 3】动态阶梯截断与去重机制"""
        sorted_items = sorted(pool.items(), key=lambda x: x[1]['score'], reverse=True)
        if not sorted_items:
            return []

        MAX_CANDIDATES = 8           
        MIN_ABS_SCORE = 0.3          
        CLIFF_TOLERANCE = 0.15       
        MAX_SAME_L2 = 4              
        
        top1_score = sorted_items[0][1]['score']
        dynamic_min_score = top1_score * 0.5  
        
        final_candidates = []
        l2_counts = {}

        for i, (code, data) in enumerate(sorted_items):
            score = data['score']
            is_force_bind = (data['source'] == "GraphRAG_Force_Bind")

            # 🛠️ 修补逻辑：图谱强制保送项，免疫一切裁员规则
            if not is_force_bind:
                if score < MIN_ABS_SCORE or score < dynamic_min_score:
                    break 
                if i > 0:
                    prev_score = sorted_items[i-1][1]['score']
                    if (prev_score - score) > CLIFF_TOLERANCE:
                        break 

                # 🛠️ 修补逻辑：多样性拦截，仅对纯向量路生效
                l2_prefix = "-".join(code.split('-')[:2]) 
                if l2_counts.get(l2_prefix, 0) >= MAX_SAME_L2:
                    continue
                l2_counts[l2_prefix] = l2_counts.get(l2_prefix, 0) + 1

            final_candidates.append((code, data))

            if len(final_candidates) >= MAX_CANDIDATES:
                break

        return final_candidates

    def _apply_wormhole_warnings(self, final_candidates: List[Tuple[str, Dict]]):
        """【阶段 4】决战圈的虫洞排查：交叉对比决赛选手，注入致命红线"""
        # O(N^2) 遍历，因为 N<=8，计算开销极小
        for i in range(len(final_candidates)):
            for j in range(i + 1, len(final_candidates)):
                code_a = final_candidates[i][0]
                code_b = final_candidates[j][0]
                
                # API 3: 调用图谱，检查两者之间是否水火不容
                rule = self.graph.check_wormhole(code_a, code_b)
                if rule:
                    final_candidates[i][1]["wormhole_warnings"].add(f"绝对互斥预警: 与 [{code_b}] 不兼容 -> {rule}")
                    final_candidates[j][1]["wormhole_warnings"].add(f"绝对互斥预警: 与 [{code_a}] 不兼容 -> {rule}")

    def _format_prompt_context(self, final_candidates: List[Tuple[str, Dict]]) -> str:
        """【阶段 5】组装高密度信息胶囊"""
        context_blocks = ["<candidates_reference>\n"]
        
        for rank, (code, data) in enumerate(final_candidates):
            score = data['score']
            name = data['name']
            features = data['features']
            source_tag = "📌 图谱特权保送" if data['source'] == "GraphRAG_Force_Bind" else f"🌐 语义匹配: {score:.2f}"
            
            block = f"### [{code}] {name} ({source_tag})\n"
            
            clean_text = re.sub(r'主要工作任务.*', '', data['text'], flags=re.DOTALL).strip()
            block += f"- 官方定义: {clean_text}\n"
            
            f_actions = ", ".join(features.get('动作', []))[:50] 
            f_objs = ", ".join(features.get('对象', []))[:50]
            f_envs = ", ".join(features.get('环境', []))[:50]
            
            block += f"- 🔍 图谱核磁扫描: 动作[{f_actions}]; 对象[{f_objs}]; 环境[{f_envs}]\n"
            
            # --- 拼装基础法理红线 ---
            redlines = []
            if features.get("是否涉公权"): redlines.append("⚠️ 涉国家公权执行")
            if features.get("是否涉临床"): redlines.append("⚠️ 涉医疗处方/临床")
            if redlines:
                block += f"- ⚖️ 法理体检: {' | '.join(redlines)}\n"
                
            # --- 拼装鉴别诊断口诀 ---
            if data['confusion_warnings']:
                block += f"- 🚨 系统级鉴别诊断 (请严格应用以下规则进行抉择):\n"
                for warning in data['confusion_warnings']:
                    block += f"   👉 {warning}\n"
                    
            # --- 拼装虫洞交叉红线 ---
            if data['wormhole_warnings']:
                block += f"- ⛔ 虫洞互斥排查 (本选项与下方其他候选项存在业务冲突):\n"
                for warning in data['wormhole_warnings']:
                    block += f"   ❌ {warning}\n"
            
            block += "\n"
            context_blocks.append(block)
            
        context_blocks.append("</candidates_reference>")
        
        final_str = "".join(context_blocks)
        if len(final_str) > 4500:
            final_str = final_str[:4500] + "\n...[为保护大模型上下文，后续备选项已截断]\n</candidates_reference>"
            
        return final_str