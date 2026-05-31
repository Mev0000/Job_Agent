import os
from openai import OpenAI

class Gemma4Client:
    def __init__(self, config):
        """
        初始化 Gemma 4 API 客户端
        :param config: 配置字典，包含 api_key, base_url, model_name 等
        """
        # 从配置中读取参数，如果没提供则使用默认的本地 vLLM 端口
        self.api_key = config.get("api_key", "EMPTY")  # vLLM 本地通常不需要鉴权
        self.base_url = config.get("base_url", "http://localhost:8000/v1")
        self.model_name = config.get("model_name", "gemma-4-31b-it") # 需要与你启动 vLLM 时的名字一致
        
        # 实例化 OpenAI 客户端
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
                # top_p=0.9 # 可选配置
            )
            
            # 返回模型生成的纯文本结果
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ 调用 Gemma 4 模型失败: {e}")
            return None