class ModelManager:
    def __init__(self):
        self.config = {
            "target_models": {
                "Shubiaobiao/DeepSeek-V3": {
                    "api_key": "your api key",
                    "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "model_id": "qwen-max"
                }
            }
        } 