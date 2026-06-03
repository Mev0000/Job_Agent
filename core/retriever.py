# core/retriever.py

import torch
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 引入重构后的图谱类
from core.graph_builder import DictGraphRAG

class AdvancedRetriever:
    def __init__(self, config: Dict[str, Any], occupation_corpus: List[Dict]):
        """
        初始化双轨安检式 RAG 检索器 (Dual-Track Auditing RAG - 级联精排完全体)
        """
        self.config = config
        self.device = "cpu"
        self.top_k_recall = config.get("retriever", {}).get("top_k_recall", 100)
        
        # ---------------- 1. 初始化 BGE-M3 粗排向量引擎 (放到 CPU) ----------------
        base_model_path = config.get("retriever", {}).get("model_path", "BAAI/bge-m3")
        
        print(f"🔄 [轨道 1-粗排] 正在加载 BGE-M3 (支持 Dense+Sparse+ColBERT) 到 CPU: {base_model_path} ...")
        self.base_model = BGEM3FlagModel(
            base_model_path, 
            use_fp16=False,      
            device=self.device   
        )
        
        self.corpus = occupation_corpus
        self.corpus_texts = [item['text'] for item in self.corpus]
        self.corpus_dict = {item['code']: item for item in self.corpus}
        
        print(f"🔄 正在为 {len(self.corpus_texts)} 条大典语料建立混合索引 (提示：CPU计算较慢，约需1-2分钟)...")
        corpus_texts = [item['text'] for item in self.corpus]

        self.corpus_embeddings = self.base_model.encode(
            corpus_texts,  # ✅ 正确：只传入纯文本列表
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
            batch_size=16
        )
        
        # ---------------- 2. 初始化 BGE-Reranker 精排引擎 (放到 CPU) ----------------
        reranker_path = config.get("retriever", {}).get("reranker_path", "BAAI/bge-reranker-v2-m3")
        self.top_k_rerank = config.get("retriever", {}).get("top_k_rerank", 15) # 精排选 Top 15 进图谱
        
        print(f"🔄 [轨道 1-精排] 正在加载 BGE-Reranker 交叉模型 到 CPU: {reranker_path} ...")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_path)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_path)
        # 🌟 修复 3：将 HuggingFace 模型挂载到 CPU (默认保持 fp32 精度)
        self.reranker_model.to(self.device)
        self.reranker_model.eval()

        # ---------------- 3. 初始化 GraphRAG 专家安检引擎 ----------------
        print(f"🔄 [轨道 2-安检] 正在加载 GraphRAG 专家安检引擎...")
        graph_dir = config.get("data", {}).get("graph_data_dir", "data/graph_tables/")
        self.graph = DictGraphRAG(data_dir=graph_dir)
        
        cache_path = config.get("data", {}).get("graph_cache_path", "data/cache/job_dict_graph.pkl")
        try:
            self.graph.load_from_disk(cache_path)
        except:
            print("⚠️ 未找到图谱缓存，触发全量动态构建...")
            self.graph.build_graph()
            self.graph.save_to_disk(cache_path)

        print("✅ 级联双轨检索引擎 (Base + Reranker + GraphRAG) 上线完毕！")

    def retrieve(self, job_name: str, job_desc: str) -> str:
        """核心调度枢纽：执行 5 阶段双轨融合召回"""
        # 阶段 1：BGE-M3 粗排 + BGE-Reranker 精排级联 (向量路)
        vector_candidates = self._get_vector_candidates(job_name, job_desc)
        
        # 阶段 2：GraphRAG 深度安检与强制绑定 (图谱路)
        audited_pool = self._audit_and_bind(vector_candidates)
        
        # 阶段 3：宁缺毋滥的断崖式阶梯截断
        final_candidates = self._cliff_cut(audited_pool)
        
        # 阶段 4：决战圈虫洞交叉排查 (Wormhole 互斥扫描)
        self._apply_wormhole_warnings(final_candidates)
        
        # 阶段 5：组装高密度信息胶囊 (喂给 LLM)
        final_prompt_context = self._format_prompt_context(final_candidates)
        
        return final_prompt_context

    def _compute_rerank_scores(self, query: str, texts: List[str], batch_size=4) -> np.ndarray:
        """原生 Reranker 批处理打分机制，防 OOM"""
        pairs = [[query, text] for text in texts]
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                inputs = self.reranker_tokenizer(
                    batch_pairs, padding=True, truncation=True, 
                    return_tensors='pt', max_length=2048
                ).to(self.device)
                
                # 获取 logits
                scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1,).float()
                all_scores.extend(scores.cpu().numpy().tolist())
                
        # Sigmoid 归一化：将 logits (-10~10) 映射到 [0, 1] 区间，完美适配后续断崖截断阈值
        scores_arr = np.array(all_scores)
        probs = 1 / (1 + np.exp(-scores_arr)) 
        return probs

    def _get_vector_candidates(self, job_name: str, job_desc: str) -> List[Dict]:
        """【阶段 1】混合级联检索：Dense+Sparse初筛 -> ColBERT提权 -> Reranker精排"""
        query_text = f"岗位名称：{job_name}。职责描述：{job_desc}"
        
        # 获取 Query 的三种向量
        query_embs = self.base_model.encode(
            [query_text], 
            return_dense=True, return_sparse=True, return_colbert_vecs=True
        )
        
        # =======================================================
        # 步骤 A: Dense (稠密) + Sparse (稀疏) 混合打分 -> 捞 Top 500
        # =======================================================
        # 计算 Dense 得分 (内积)
        dense_scores = query_embs['dense_vecs'][0] @ self.corpus_embeddings['dense_vecs'].T
        
        # 计算 Sparse 得分 (词汇匹配)
        sparse_scores = np.array([
            self.base_model.compute_lexical_matching_score(query_embs['lexical_weights'][0], doc_weight) 
            for doc_weight in self.corpus_embeddings['lexical_weights']
        ])
        
        # 归一化 (Min-Max)
        dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
        sparse_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-8)
        
        # 混合基础得分 0.5 + 0.5
        base_scores = 0.5 * dense_norm + 0.5 * sparse_norm
        
        # 选出基础得分最高的 Top 500 索引
        top500_indices = np.argsort(base_scores)[::-1][:500]
        
        # =======================================================
        # 步骤 B: ColBERT 细粒度提权 -> 截取 Top 100
        # =======================================================
        colbert_scores = []
        for idx in top500_indices:
            # 仅对 Top 500 计算高开销的 ColBERT 分数
            score = self.base_model.colbert_score(
                query_embs['colbert_vecs'][0], 
                self.corpus_embeddings['colbert_vecs'][idx]
            )
            colbert_scores.append(score)
            
        colbert_scores = np.array(colbert_scores)
        colbert_norm = (colbert_scores - colbert_scores.min()) / (colbert_scores.max() - colbert_scores.min() + 1e-8)
        
        # 获取 Top 500 对应的原本 base_norm 基础分
        top500_base_norm = base_scores[top500_indices]
        
        # 执行非线性拉伸提权公式
        final_recall_scores = (0.5 * top500_base_norm + 0.5 * colbert_norm) ** 1.2
        
        # 从这 500 个里，再挑出综合得分最高的前 top_k_recall (100) 个，进入 Reranker
        top100_local_indices = np.argsort(final_recall_scores)[::-1][:self.top_k_recall]
        top_k_indices = [top500_indices[i] for i in top100_local_indices]
        
        recall_candidates = [self.corpus[idx] for idx in top_k_indices]
        recall_texts = [item['text'] for item in recall_candidates]
        
        # =======================================================
        # 步骤 C: Reranker 交叉注意力精排
        # =======================================================
        rerank_probs = self._compute_rerank_scores(query_text, recall_texts)
        
        # 获取精排后的 Top 15 索引，准备交接给 GraphRAG 安检
        best_indices = np.argsort(rerank_probs)[::-1][:self.top_k_rerank] # 默认 15
        
        final_vector_candidates = []
        for idx in best_indices:
            orig_item = recall_candidates[idx]
            final_vector_candidates.append({
                "code": orig_item['code'],
                "name": orig_item.get('name', '未知职务'),
                "text": orig_item['text'],
                "score": float(rerank_probs[idx])
            })
            
        return final_vector_candidates

    def _audit_and_bind(self, vector_candidates: List[Dict]) -> Dict[str, Dict]:
        """【阶段 2】图谱特征注入与易混淆强制绑定"""
        audited_pool = {}
        for cand in vector_candidates:
            code = cand['code']
            if code in audited_pool: continue
                
            features = self.graph.get_job_features(code)
            if not features: continue 
                
            audited_pool[code] = {
                "source": "Vector_Search", "score": cand['score'],
                "name": cand['name'], "text": cand['text'],
                "features": features, "confusion_warnings": set(), "wormhole_warnings": set()
            }
            
            confused_codes = self.graph.get_confused_codes(code)
            for c_code in confused_codes:
                rule_text = self.graph.get_confusion_text(code, c_code)
                
                if c_code not in audited_pool:
                    c_features = self.graph.get_job_features(c_code)
                    c_corpus_data = self.corpus_dict.get(c_code)
                    if c_features and c_corpus_data:
                        audited_pool[c_code] = {
                            "source": "GraphRAG_Force_Bind", 
                            "score": cand['score'] - 0.001, # 绑定分数
                            "name": c_corpus_data.get('name', '未知职务'),
                            "text": c_corpus_data['text'],
                            "features": c_features,
                            "confusion_warnings": set(), "wormhole_warnings": set()
                        }
                
                audited_pool[code]["confusion_warnings"].add(f"与 [{c_code}] 易混淆：{rule_text}")
                if c_code in audited_pool:
                    audited_pool[c_code]["confusion_warnings"].add(f"与 [{code}] 易混淆：{rule_text}")

        return audited_pool

    def _cliff_cut(self, pool: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        """【阶段 3】动态阶梯截断与去重机制"""
        sorted_items = sorted(pool.items(), key=lambda x: x[1]['score'], reverse=True)
        if not sorted_items: return []

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

            if not is_force_bind:
                if score < MIN_ABS_SCORE or score < dynamic_min_score: break 
                if i > 0 and (sorted_items[i-1][1]['score'] - score) > CLIFF_TOLERANCE: break 

                l2_prefix = "-".join(code.split('-')[:2]) 
                if l2_counts.get(l2_prefix, 0) >= MAX_SAME_L2: continue
                l2_counts[l2_prefix] = l2_counts.get(l2_prefix, 0) + 1

            final_candidates.append((code, data))
            if len(final_candidates) >= MAX_CANDIDATES: break

        return final_candidates

    def _apply_wormhole_warnings(self, final_candidates: List[Tuple[str, Dict]]):
        """【阶段 4】决战圈的虫洞排查"""
        for i in range(len(final_candidates)):
            for j in range(i + 1, len(final_candidates)):
                code_a = final_candidates[i][0]
                code_b = final_candidates[j][0]
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
            source_tag = "📌 图谱特权保送" if data['source'] == "GraphRAG_Force_Bind" else f"🌐 语义精排置信度: {score:.2f}"
            
            block = f"### [{code}] {name} ({source_tag})\n"
            clean_text = re.sub(r'主要工作任务.*', '', data['text'], flags=re.DOTALL).strip()
            block += f"- 官方定义: {clean_text}\n"
            
            f_actions = ", ".join(features.get('动作', []))[:50] 
            f_objs = ", ".join(features.get('对象', []))[:50]
            f_envs = ", ".join(features.get('环境', []))[:50]
            block += f"- 🔍 图谱核磁扫描: 动作[{f_actions}]; 对象[{f_objs}]; 环境[{f_envs}]\n"
            
            redlines = []
            if features.get("是否涉公权"): redlines.append("⚠️ 涉国家公权执行")
            if features.get("是否涉临床"): redlines.append("⚠️ 涉医疗处方/临床")
            if redlines: block += f"- ⚖️ 法理体检: {' | '.join(redlines)}\n"
                
            if data['confusion_warnings']:
                block += f"- 🚨 系统级鉴别诊断 (请严格应用以下规则进行抉择):\n"
                for warning in data['confusion_warnings']: block += f"   👉 {warning}\n"
                    
            if data['wormhole_warnings']:
                block += f"- ⛔ 虫洞互斥排查 (本选项与下方其他候选项存在业务冲突):\n"
                for warning in data['wormhole_warnings']: block += f"   ❌ {warning}\n"
            
            block += "\n"
            context_blocks.append(block)
            
        context_blocks.append("</candidates_reference>")
        final_str = "".join(context_blocks)
        if len(final_str) > 4500:
            final_str = final_str[:4500] + "\n...[为保护大模型上下文，后续备选项已截断]\n</candidates_reference>"
            
        return final_str