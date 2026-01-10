import logging
import os
import requests
import json
# 建议使用 faster-whisper，因为它在本地运行极快，且不容易断连
from faster_whisper import WhisperModel

# 导入配置管理器
from src.utils.config import config_manager

logger = logging.getLogger(__name__)

class VoiceService:
    def __init__(self, model_size="base"):
        # 初始化本地模型（faster-whisper）
        # 第一次运行会下载模型（约 150MB）
        # 如果有显卡会自动用 CUDA，没有则用 CPU
        self.local_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        # 初始化 Qwen 语音模型配置
        self.qwen_api_key = config_manager.get("ai.api_key", "")
        self.qwen_api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/audio-generation/text-to-speech"
        self.qwen_asr_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/speech-recognition/audio-to-text"
        self.qwen_model = "qwen-turbo"
        
        # 优先使用本地模型，Qwen 作为备选
        self.use_local_model = True

    def transcribe(self, audio_path):
        """将音频文件转换为文字
        优先使用本地模型，失败后使用 Qwen API
        """
        if not audio_path or not os.path.exists(audio_path):
            return ""
        
        # 1. 优先尝试本地模型
        if self.use_local_model:
            try:
                segments, info = self.local_model.transcribe(audio_path, beam_size=5)
                text = "".join([segment.text for segment in segments])
                logger.info(f"本地语音识别成功: {text}")
                return text
            except Exception as e:
                logger.error(f"本地语音识别失败，尝试使用 Qwen API: {e}")
                # 本地模型失败，尝试使用 Qwen API
        
        # 2. 尝试使用 Qwen API
        if self.qwen_api_key:
            try:
                return self._qwen_transcribe(audio_path)
            except Exception as e:
                logger.error(f"Qwen 语音识别失败: {e}")
                return f"识别出错: {str(e)}"
        
        # 3. 所有模型都不可用
        return "语音识别服务不可用，请检查配置"
        
    def _qwen_transcribe(self, audio_path):
        """使用 Qwen API 进行语音识别"""
        headers = {
            "Content-Type": "multipart/form-data",
            "Authorization": f"Bearer {self.qwen_api_key}"
        }
        
        # 读取音频文件
        with open(audio_path, "rb") as f:
            files = {
                "file": f
            }
            
            # 调用 Qwen API
            response = requests.post(
                self.qwen_asr_url,
                headers=headers,
                files=files,
                data={"model": "paraformer-v2", "language": "zh-CN"}
            )
        
        # 处理响应
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 200 and result.get("output"):
                text = result["output"].get("text", "")
                logger.info(f"Qwen 语音识别成功: {text}")
                return text
            else:
                logger.error(f"Qwen API 返回错误: {result}")
                return f"Qwen API 错误: {result.get('message', '未知错误')}"
        else:
            logger.error(f"Qwen API 请求失败: {response.status_code}, {response.text}")
            return f"Qwen API 请求失败: {response.status_code}"

# 单例模式
_voice_instance = None
def get_voice_service():
    global _voice_instance
    if _voice_instance is None:
        _voice_instance = VoiceService()
    return _voice_instance
