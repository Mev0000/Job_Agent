# core/retriever.py

import json
import os
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

        # ---------------- 3.5 加载蒸馏 7D 图谱数据（从 graph_nodes.json 注入全量七维属性）----------------
        self._last_hop1_pool = {}  # Hop-1 全量候选池，供跨 L2 对照注入使用
        self._graph_7d = {}
        cache_dir = os.path.dirname(config.get("data", {}).get("graph_cache_path", "data/cache/job_dict_graph.pkl"))
        nodes_7d_path = os.path.join(cache_dir, "graph_nodes.json")
        if os.path.exists(nodes_7d_path):
            with open(nodes_7d_path, 'r', encoding='utf-8') as f:
                self._graph_7d = json.load(f)
            print(f"✅ 加载蒸馏 7D 图谱: {len(self._graph_7d)} 个节点（Step 4 比对升级为 ground truth 模式）")
        else:
            print(f"⚠️ 未找到 {nodes_7d_path}，Step 4 比对降级为 LLM 盲猜模式")

        print("✅ 级联双轨检索引擎 (Base + Reranker + GraphRAG) 上线完毕！")

    def retrieve(self, job_name: str, job_desc: str) -> str:
        """核心调度枢纽：执行 5 阶段双轨融合召回"""
        # 阶段 1：BGE-M3 粗排 + BGE-Reranker 精排级联 (向量路)
        vector_candidates = self._get_vector_candidates(job_name, job_desc)
        
        # 阶段 2：GraphRAG 深度安检与强制绑定 (图谱路)
        audited_pool = self._audit_and_bind(vector_candidates, job_name, job_desc)
        
        # 🔧 P0: 保存 Hop-1 全量候选池，供后续跨 L2 对照注入使用
        self._last_hop1_pool = audited_pool
        
        # 阶段 3：宁缺毋滥的断崖式阶梯截断
        final_candidates = self._cliff_cut(audited_pool)
        
        print(f"  🔍 [检索调试] 审计池: {len(audited_pool)} 个候选 | 截断后: {len(final_candidates)} 个候选")
        if len(final_candidates) == 0:
            # 🔧 修复：sorted_items 是 _cliff_cut() 局部变量，此处无法访问，改用 audited_pool 诊断
            top5 = sorted(audited_pool.items(), key=lambda x: x[1].get('score', 0), reverse=True)[:5]
            top5_scores = [f"{code}:{data.get('score', 0):.3f}" for code, data in top5]
            print(f"  ⚠️ [检索调试] 所有候选被截断！Top5 分数: {', '.join(top5_scores)}")
            print(f"  ⚠️ [检索调试] MIN_ABS_SCORE: 0.1 (局部变量，非实例属性)")
        
        # 阶段 4：决战圈虫洞交叉排查 (Wormhole 互斥扫描)
        self._apply_wormhole_warnings(final_candidates)
        
        # 阶段 5：组装高密度信息胶囊 (喂给 LLM)
        final_prompt_context = self._format_prompt_context(final_candidates)
        
        # P1-A（Prompt注入）：检测口诀强制命中，注入提示给LLM
        rule_forced_in_final = []
        for code, data in audited_pool.items():
            if code not in final_candidates:
                continue
            warnings = data.get("confusion_warnings", set())
            if any("严禁" in w or "必须" in w or "只能" in w or "强制" in w for w in warnings):
                rule_forced_in_final.append(code)
        if rule_forced_in_final:
            final_prompt_context += "\n⚠️ 口诀强制命中：" + "、".join(rule_forced_in_final) + "\n"
        
        # P1-B（跨 L2 对照注入 — 增强版）：防止 Hop-1 过早定谳选错大类
        # 从 Hop-1 全局审计池中提取其他 L2 高分候选供 LLM 对照
        current_l2_prefixes = list(set("-".join(code.split("-")[:2]) for code, _ in final_candidates))
        cross_l2_context = self._inject_cross_l2_challengers(current_l2_prefixes)
        if cross_l2_context:
            final_prompt_context += (
                "\n\n⚠️⚠️⚠️ 【Hop-1 关键提醒：跨大类对照候选已注入】 ⚠️⚠️⚠️\n"
                "以下来自其他大类的候选人在 Hop-1 初始检索中获得了与当前候选相近的分数。\n"
                "请在 Step 1 大类定位时，认真对比这些跨类候选人的劳动事实，\n"
                "确认当前候选的 L2 大类方向是否正确，避免 Hop-1 过早定谳出错！\n"
            )
            final_prompt_context += cross_l2_context

        return final_prompt_context

    def retrieve_scoped(self, job_name: str, job_desc: str, l2_prefixes: List[str]) -> str:
        """【REQ_L2 兜底】在指定 L2 范围内做二次 Reranker 向量召回 + 置信度门控
        
        当 LLM 触发 REQ_L2 选定大类后，直接用 Reranker 对 L2 范围内所有候选做交叉打分。
        L2 通常 10-80 条，速度快。返回 Top 8 候选的完整 7D 格式化上下文。
        
        🌟 置信度门控：若该 L2 内 Top 3 Reranker 平均分低于阈值，
        系统会在返回的上下文中注入「低置信度警报」，提示 LLM 该 L2 可能选错。
        用于解决「LLM 选错 L2 → scoped 召回全错 → LLM 更自信地选错」的死亡螺旋。
        
        Args:
            job_name: 岗位名称
            job_desc: 岗位职责描述
            l2_prefixes: L2 前缀列表，如 ["4-01"]
            
        Returns:
            格式化后的候选上下文字符串（含置信度信号）
        """
        query_text = f"岗位名称：{job_name}。职责描述：{job_desc}"
        
        # 1. 筛选 L2 范围内的候选
        scoped_candidates = []
        for item in self.corpus:
            for prefix in l2_prefixes:
                if item['code'].startswith(prefix + '-'):
                    scoped_candidates.append(item)
                    break
        
        if not scoped_candidates:
            return ""
        
        # 2. 用 Reranker 对所有范围内候选打分
        texts = [item['text'] for item in scoped_candidates]
        scores = self._compute_rerank_scores(query_text, texts)
        
        # 3. 取 Top 8 高置信度候选
        top_k = min(8, len(scoped_candidates))
        best_indices = np.argsort(scores)[::-1][:top_k]
        
        # ── 置信度门控：Top 3 平均分 ──
        top3_avg = float(np.mean(scores[best_indices[:min(3, top_k)]]))
        top1_score = float(scores[best_indices[0]])
        low_confidence = top3_avg < 0.3  # 阈值：经验值，后续可根据实际数据调整
        
        vector_candidates = []
        for idx in best_indices:
            item = scoped_candidates[idx]
            vector_candidates.append({
                "code": item['code'],
                "name": item.get('name', '未知职务'),
                "text": item['text'],
                "score": float(scores[idx])
            })
        
        # 4. 走标准管线：图谱安检 → 截断 → 虫洞 → 格式化
        audited_pool = self._audit_and_bind(vector_candidates, job_name, job_desc)
        final_candidates = self._cliff_cut(audited_pool)
        self._apply_wormhole_warnings(final_candidates)
        context = self._format_prompt_context(final_candidates)
        
        # ── 5. 置信度信号注入 ──
        signal_header = ""
        if low_confidence:
            signal_header = (
                f"\n\n🚨🚨🚨 【系统置信度警报 — 请认真阅读】🚨🚨🚨\n"
                f"Reranker 对该 L2({l2_prefixes})范围内所有候选做了交叉注意力打分：\n"
                f"  - 最高分: {top1_score:.3f}\n"
                f"  - Top 3 平均分: {top3_avg:.3f}\n"
                f"  - 阈值: 0.30\n\n"
                f"⚠️ 该 L2 大类下候选与你的岗位描述匹配度【整体偏低】，原因极可能是：\n"
                f"  1. 你选错了二级大类（当前在 {l2_prefixes}，正确大类可能在其他地方）\n"
                f"  2. 或者该岗位确实是一个边缘/跨类岗位\n\n"
                f"🔴 强制要求：请在 Step 3 独立假说中明确讨论「当前选 {l2_prefixes} 是否正确」。\n"
                f"   如果确信选错，立即输出 action=\"REQ_L2\" 换大类，严禁凑合 FINALIZE。\n"
                f"🚨🚨🚨 【警报结束】🚨🚨🚨"
            )
        else:
            signal_header = (
                f"\n\n📊 【系统信息：置信度信号】\n"
                f"Reranker 对该 L2 范围内 Top 3 平均分: {top3_avg:.3f}（≥ 阈值 0.30，置信度正常）\n"
                f"最高分候选: {top1_score:.3f}"
            )
        
        # ── 6. 跨 L2 对照候选注入（防止选错大类的死亡螺旋）──
        cross_l2_context = self._inject_cross_l2_challengers(l2_prefixes)
        
        return signal_header + "\n\n" + context + cross_l2_context

    def _inject_cross_l2_challengers(self, target_l2_prefixes: List[str]) -> str:
        """【P0 跨 L2 对照注入】从 Hop-1 全局池中提取其他 L2 的高分候选，注入到 scoped 召回上下文
        
        解决"LLM 选错 L2 → scoped 召回全错 → LLM 更自信地选错"的死亡螺旋。
        从 Hop-1 全局候选池中捞出不属于当前 L2 但分数较高的候选，
        附带完整 7D 特征卡，让 LLM 在 Step 4 做跨类七维比对。
        
        Args:
            target_l2_prefixes: 当前选中的 L2 前缀列表，如 ["4-01"]
            
        Returns:
            格式化后的跨 L2 对照候选上下文字符串，无候选时返回空字符串
        """
        if not self._last_hop1_pool:
            return ""
        
        # 收集所有非目标 L2 的候选
        cross_l2_candidates = []
        for code, data in self._last_hop1_pool.items():
            l2_prefix = "-".join(code.split("-")[:2])
            if l2_prefix not in target_l2_prefixes:
                # 补齐 7D（如果还未合并）
                features = data.get("features", {})
                nd7 = self._graph_7d.get(code)
                if nd7 and "deliverables" not in features:
                    features["deliverables"] = nd7.get("deliverables", [])
                    features["served_population"] = nd7.get("served_population", [])
                    features["role_level"] = nd7.get("role_level", "")
                    features["main_kpi"] = nd7.get("main_kpi", "")
                
                cross_l2_candidates.append({
                    "code": code,
                    "l2": l2_prefix,
                    "score": data.get("score", 0),
                    "name": data.get("name", "未知"),
                    "features": features,
                    "source": data.get("source", "Vector_Search")
                })
        
        if not cross_l2_candidates:
            return ""
        
        # 按分数排序，取 Top 3（从2扩至3，增强对照力度）
        cross_l2_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_challengers = cross_l2_candidates[:3]
        
        # 格式化输出
        blocks = [
            "\n\n🌐🌐🌐 【跨 L2 对照候选 — 来自 Hop-1 全局池的其他大类高分候选】🌐🌐🌐",
            f"⚠️ 以下候选来自【非 {', '.join(target_l2_prefixes)}】的其他大类，但在 Hop-1 全局检索中得分较高。",
            "请在 Step 4 七维比对时，也对照检查以下候选人，确认当前 L2 选择是否正确：\n"
        ]
        
        for rank, c in enumerate(top_challengers, 1):
            f = c["features"]
            a = ", ".join(f.get("core_actions", []) or f.get("动作", []))[:80]
            d = ", ".join(f.get("deliverables", []) or [])[:80]
            s = ", ".join(f.get("served_population", []) if isinstance(f.get("served_population"), list) else ([f.get("served_population", "")] if f.get("served_population") else []))[:60]
            r = f.get("role_level", "") or ""
            
            blocks.append(
                f"  [{rank}] [{c['code']}] {c['name']} (L2={c['l2']}, 分数={c['score']:.3f}, 来源={c['source']})\n"
                f"      核心动作: {a}\n"
                f"      交付物:   {d}\n"
                f"      服务对象: {s}\n"
                f"      责任层级: {r}"
            )
        
        blocks.append("\n🔴 强制要求：在 Step 3 独立假说中明确讨论上述对照候选是否更匹配。如发现当前 L2 错误，立即 REQ_L2 换路！")
        blocks.append("🌐🌐🌐 【跨 L2 对照候选结束】🌐🌐🌐\n")
        
        return "\n".join(blocks)

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

    def _audit_and_bind(self, vector_candidates: List[Dict], job_name: str, job_desc: str) -> Dict[str, Dict]:
        """【阶段 2】图谱特征注入与易混淆强制绑定
        
        P1 优化：Force-bind 候选使用 Reranker 独立打分，
        而非简单继承原始候选分数（避免分数严重失真）
        """
        audited_pool = {}
        query_text = f"岗位名称：{job_name}。职责描述：{job_desc}"

        for cand in vector_candidates:
            code = cand['code']
            if code in audited_pool: continue
                
            features = self.graph.get_job_features(code)
            if not features: continue 
            
            # 🔧 P0-FIX：合并蒸馏 7D（从 graph_nodes.json），补齐 deliverables / served_population / role_level
            nd7 = self._graph_7d.get(code)
            if nd7:
                features["core_actions"] = nd7.get("core_actions", [])
                features["objects"] = nd7.get("objects", [])
                features["deliverables"] = nd7.get("deliverables", [])
                features["main_kpi"] = nd7.get("main_kpi", "")
                features["environment"] = nd7.get("environment", [])
                features["served_population"] = nd7.get("served_population", [])
                features["role_level"] = nd7.get("role_level", "")
                features["category"] = nd7.get("category", "")
                features["Is_Government"] = nd7.get("Is_Government", features.get("是否涉公权", False))
                features["Is_Medical_Clinical"] = nd7.get("Is_Medical_Clinical", features.get("是否涉临床", False))
                
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
                    # 🔧 同样为 Force-Bind 候选补齐 7D
                    c_nd7 = self._graph_7d.get(c_code)
                    if c_nd7:
                        c_features["core_actions"] = c_nd7.get("core_actions", [])
                        c_features["objects"] = c_nd7.get("objects", [])
                        c_features["deliverables"] = c_nd7.get("deliverables", [])
                        c_features["main_kpi"] = c_nd7.get("main_kpi", "")
                        c_features["environment"] = c_nd7.get("environment", [])
                        c_features["served_population"] = c_nd7.get("served_population", [])
                        c_features["role_level"] = c_nd7.get("role_level", "")
                        c_features["category"] = c_nd7.get("category", "")
                        c_features["Is_Government"] = c_nd7.get("Is_Government", c_features.get("是否涉公权", False))
                        c_features["Is_Medical_Clinical"] = c_nd7.get("Is_Medical_Clinical", c_features.get("是否涉临床", False))
                    c_corpus_data = self.corpus_dict.get(c_code)
                    if c_features and c_corpus_data:
                        # P1 优化：使用 Reranker 对 JD 和该易混对手单独打分
                        c_text = c_corpus_data['text']
                        with torch.no_grad():
                            inputs = self.reranker_tokenizer(
                                [[query_text, c_text]],  # ✅ 修复：传入 List[List[str, str]] 格式
                                padding=True, truncation=True,
                                return_tensors='pt', max_length=2048
                            ).to(self.device)
                            logits = self.reranker_model(**inputs, return_dict=True).logits
                            # logits 形状: (1, 1) 或 (1, num_labels)，取第一个元素的第一个 logit
                            c_score = float(torch.sigmoid(logits[0][0]))
                        
                        audited_pool[c_code] = {
                            "source": "GraphRAG_Force_Bind", 
                            "score": c_score,  # 独立打分，非继承
                            "name": c_corpus_data.get('name', '未知职务'),
                            "text": c_text,
                            "features": c_features,
                            "confusion_warnings": set(), "wormhole_warnings": set()
                        }
                
                audited_pool[code]["confusion_warnings"].add(f"与 [{c_code}] 易混淆：{rule_text}")
                if c_code in audited_pool:
                    audited_pool[c_code]["confusion_warnings"].add(f"与 [{code}] 易混淆：{rule_text}")
        
        # P1 优化：对所有 Force_Bind 候选做一轮 Reranker 独立打分（修正分数失真）
        force_bind_codes = [c for c, d in audited_pool.items() if d['source'] == "GraphRAG_Force_Bind"]
        if force_bind_codes:
            rerank_texts = [audited_pool[c]['text'] for c in force_bind_codes]
            # 使用已有的 _compute_rerank_scores 方法，避免重复代码和潜在错误
            rerank_probs = self._compute_rerank_scores(query_text, rerank_texts)
            for c, s in zip(force_bind_codes, rerank_probs):
                audited_pool[c]['score'] = float(s)

        return audited_pool

    def _cliff_cut(self, pool: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        """【阶段 3】动态阶梯截断与去重机制
        
        P1 优化：Force-Bind 候选最多占 4/8 席，防止易混对手挤占向量精排结果
        P2 优化：CLIFF_TOLERANCE 从 0.15 → 0.20，防止高置信度第一名直接截断后续对照候选
                 新增 MIN_CANDIDATES=3 保底，截断后候选数不足3时自动补入次高分候选（忽略断崖）
        """
        sorted_items = sorted(pool.items(), key=lambda x: x[1]['score'], reverse=True)
        if not sorted_items: return []

        MAX_CANDIDATES = 8           
        MIN_ABS_SCORE = 0.1          # 🔧 修复：从 0.3 降到 0.1，防止高质量候选被过滤
        CLIFF_TOLERANCE = 0.35       # 🔧 P3 优化：从 0.20 → 0.35，进一步放宽断崖容忍度，防止第一名截断所有对照候选
        MIN_CANDIDATES = 5           # 🔧 P2 优化：截断后最少保留 5 个候选供 LLM 对照（从3提升）
        MAX_SAME_L2 = 4
        MAX_FORCE_BIND = 4  # P1 优化：Force-Bind 最多占 4/8 席

        top1_score = sorted_items[0][1]['score']
        dynamic_min_score = top1_score * 0.5  

        final_candidates = []
        l2_counts = {}
        force_bind_count = 0

        for i, (code, data) in enumerate(sorted_items):
            score = data['score']
            is_force_bind = (data['source'] == "GraphRAG_Force_Bind")

            # P1 优化：Force-Bind 席位上限检查
            if is_force_bind and force_bind_count >= MAX_FORCE_BIND:
                continue

            if not is_force_bind:
                if score < MIN_ABS_SCORE or score < dynamic_min_score: break 
                if i > 0 and (sorted_items[i-1][1]['score'] - score) > CLIFF_TOLERANCE: break 

                l2_prefix = "-".join(code.split('-')[:2]) 
                if l2_counts.get(l2_prefix, 0) >= MAX_SAME_L2: continue
                l2_counts[l2_prefix] = l2_counts.get(l2_prefix, 0) + 1

            if is_force_bind:
                force_bind_count += 1

            final_candidates.append((code, data))
            if len(final_candidates) >= MAX_CANDIDATES: break

        # 🔧 P2 优化：MIN_CANDIDATES 保底机制
        # 若截断后候选数不足 MIN_CANDIDATES，从剩余 sorted_items 中按分数补入（忽略断崖限制，仅保留绝对下限）
        if len(final_candidates) < MIN_CANDIDATES:
            existing_codes = {code for code, _ in final_candidates}
            for code, data in sorted_items:
                if len(final_candidates) >= MIN_CANDIDATES:
                    break
                if code in existing_codes:
                    continue
                score = data['score']
                is_force_bind = (data['source'] == "GraphRAG_Force_Bind")
                if not is_force_bind and score < MIN_ABS_SCORE:
                    break  # 绝对下限仍然生效
                final_candidates.append((code, data))
                existing_codes.add(code)
                print(f"  🔄 [候选池保底] 补入第{len(final_candidates)}个候选: [{code}] score={score:.3f}")

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
        """【阶段 5】组装高密度信息胶囊（展示全量 7D 特征）"""
        context_blocks = ["<candidates_reference>\n"]
        MAX_CANDIDATES = 8

        for rank, (code, data) in enumerate(final_candidates):
            score = data['score']
            name = data['name']
            features = data['features']
            is_force = data['source'] == "GraphRAG_Force_Bind"
            source_tag = "📌 图谱特权保送" if is_force else f"🌐 语义精排置信度: {score:.2f}"

            block = f"### [{code}] {name} ({source_tag})\n"
            clean_text = re.sub(r'主要工作任务.*', '', data['text'], flags=re.DOTALL).strip()
            block += f"- 官方定义: {clean_text}\n"

            # ── 分层展示策略 ──
            # 前 3 条：全量 7D 展示（LLM 需要精细比对）
            # 候选 4~8：精简展示（仅名称+定义+核心动作+交付物，口诀如有则保留）
            if rank < 3:
                # 全量 7D
                a = ", ".join(features.get('core_actions', []) or features.get('动作', []))[:100]
                o = ", ".join(features.get('objects', []) or features.get('对象', []))[:100]
                d = ", ".join(features.get('deliverables', []) or features.get('交付物', []))[:100]
                e = ", ".join(features.get('environment', []) or features.get('环境', []))[:100]
                k = features.get('main_kpi', '') or ''
                s = ", ".join(features.get('served_population', []) if isinstance(features.get('served_population'), list) else ([features.get('served_population', '')] if features.get('served_population') else []))[:100]
                r = features.get('role_level', '') or ''

                block += f"- 🔍 7D 特征扫描:\n"
                block += f"  核心动作: {a}\n"   if a else ""
                block += f"  交付物:   {d}\n"   if d else ""
                block += f"  作用对象: {o}\n"   if o else ""
                block += f"  工作环境: {e}\n"   if e else ""
                block += f"  核心 KPI: {k}\n"   if k else ""
                block += f"  服务对象: {s}\n"   if s else ""
                block += f"  责任层级: {r}\n"   if r else ""
            else:
                # 精简模式：只展示最关键的诊断信息
                a_short = ", ".join(features.get('core_actions', []) or features.get('动作', []))[:60]
                d_short = ", ".join(features.get('deliverables', []) or features.get('交付物', []))[:60]
                block += f"- 🔍 核心动作: {a_short}  |  交付物: {d_short}\n"

            # 法理红线（全量展示，兼容新旧字段名）
            redlines = []
            if features.get("是否涉公权") or features.get("Is_Government"): redlines.append("⚠️ 涉国家公权执行")
            if features.get("是否涉临床") or features.get("Is_Medical_Clinical"): redlines.append("⚠️ 涉医疗处方/临床")
            if redlines: block += f"- ⚖️ 法理体检: {' | '.join(redlines)}\n"

            # 易混淆口诀（全量展示）
            if data['confusion_warnings']:
                block += f"- 🚨 系统级鉴别诊断 (请严格应用以下规则进行抉择):\n"
                for warning in data['confusion_warnings']: block += f"   👉 {warning}\n"

            # 虫洞互斥（全量展示）
            if data['wormhole_warnings']:
                block += f"- ⛔ 虫洞互斥排查 (本选项与下方其他候选项存在业务冲突):\n"
                for warning in data['wormhole_warnings']: block += f"   ❌ {warning}\n"

            block += "\n"
            context_blocks.append(block)

        context_blocks.append("</candidates_reference>")
        return "".join(context_blocks)