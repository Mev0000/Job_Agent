import os
from openai import OpenAI

class Gemma4Client:
    def __init__(self, config):
        """
        初始化 Gemma 4 API 客户端
        """
        # 获取 llm 配置块，如果找不到再用空字典兜底
        llm_config = config.get("llm", {})
        
        # 从 llm_config 中安全读取配置
        self.api_key = llm_config.get("api_key", "ollama")  
        self.base_url = llm_config.get("base_url", "http://127.0.0.1:11434/v1")
        self.model_name = llm_config.get("model_name", "gemma4:31b") 
        
        print(f"\n[🔌 LLM 探针] 当前正试图连接的 API 地址是: {self.base_url}，模型名: {self.model_name}")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate(self, system_prompt, user_prompt, temperature=0.15, max_tokens=4096):
        """
        向 Gemma 4 发送请求并获取回复
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # 开启思考模式 (Gemma 4 特性)，我们把温度调低以保证分类逻辑的严谨性
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9 # 可选配置
            )
            
            # 返回模型生成的纯文本结果
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ 调用 Gemma 4 模型失败: {e}")
            return None