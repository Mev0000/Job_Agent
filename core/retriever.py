# core/retriever.py

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class BGERetriever:
    def __init__(self, config, occupation_corpus):
        """
        初始化 BGE-M3 检索器
        :param config: 全局配置字典
        :param occupation_corpus: 职业大典的语料列表，格式如 [{"code": "4-01-02", "text": "营销员: 从事市场推销..."}]
        """
        model_path = config.get("model_path", "BAAI/bge-m3")
        self.top_k = config.get("top_k", 10)
        
        print(f"🔄 正在加载 BGE-M3 向量模型: {model_path} ...")
        # 自动调用 GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=device)
        
        self.corpus = occupation_corpus
        self.corpus_texts = [item['text'] for item in self.corpus]
        
        print(f"🔄 正在为 {len(self.corpus_texts)} 条职业大典语料建立向量索引 (离线化后可缓存)...")
        # 实际生产中，这里的 corpus_embeddings 应该被保存为 .npy 或存在 Faiss 中直接读取
        self.corpus_embeddings = self.model.encode(self.corpus_texts, normalize_embeddings=True, show_progress_bar=True)
        print("✅ 向量库加载完毕！")

    def retrieve(self, job_name, job_desc):
        """
        根据输入 JD 检索最匹配的 Top-K 职业
        返回格式化后的 Context 字符串，直接供 LLM 阅读
        """
        query_text = f"岗位名称：{job_name}。职责描述：{job_desc}"
        query_embedding = self.model.encode([query_text], normalize_embeddings=True)[0]
        
        # 计算余弦相似度
        similarities = np.dot(self.corpus_embeddings, query_embedding)
        
        # 获取 Top-K 索引
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        # 组装返回给 Gemma 4 的参考上下文
        context_lines = []
        for rank, idx in enumerate(top_indices):
            code = self.corpus[idx]['code']
            text_desc = self.corpus[idx]['text']
            score = similarities[idx]
            context_lines.append(f"[{rank+1}] 代码: {code} | 描述: {text_desc} | 向量相似度: {score:.4f}")
            
        return "\n".join(context_lines)