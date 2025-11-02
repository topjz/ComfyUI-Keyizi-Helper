import torch
import os
import io
import re
from comfy.utils import save_audio
from nodes import Node

class UploadToQiniu(Node):
    """
    一个简单的自定义节点示例：对输入张量进行缩放
    """
    # 节点类别（在UI左侧菜单中的分类）
    CATEGORY = "Keyizi Helper"
    FUNCTION = "process_audio"

    # 输入参数定义
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),  # 接收TTS的音频数据对象
                "subtitle": ("STRING", {"default": ""}),  # 接收TTS的字幕
                "seed": ("INT", {"default": 0}),  # 接收TTS的seed
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": ""}),  # 可选前缀
            }
        }

    # 输出类型定义
    RETURN_TYPES = ()  # 输出张量
    RETURN_NAMES = ("fileName",)  # 输出名称（UI中显示）

    def process_audio(self, audio, subtitle, seed, filename_prefix=""):
        # 1. 解包音频数据（tensor + 采样率，无需保存到本地）
        if not isinstance(audio, tuple) or len(audio) < 2:
            raise ValueError("无效的音频数据格式，需为 (tensor, sample_rate)")
        audio_tensor, sample_rate = audio

        # 2. 生成目标文件名（用于七牛云存储的文件名）
        filename = self.generate_filename(subtitle, filename_prefix, seed)

        # 3. 关键优化：将音频张量直接转为二进制流（无需本地保存）
        audio_bytes = self.tensor_to_audio_bytes(audio_tensor, sample_rate)

        # 4. 直接用二进制流上传七牛云（跳过本地文件保存）
        self.upload_to_qiniu(audio_bytes, filename)

        return ()

    def generate_filename(self, subtitle, prefix, seed):
        """生成七牛云存储的文件名"""
        if subtitle:
            safe_subtitle = re.sub(r'[^\w\u4e00-\u9fa5,.!? ]', '_', subtitle)
            safe_subtitle = safe_subtitle.strip().replace(' ', '_')[:20]
            return f"{safe_subtitle}_{seed}.wav"
        return f"{prefix}_{seed}.wav" if prefix else f"tts_audio_{seed}.wav"

    def tensor_to_audio_bytes(self, audio_tensor, sample_rate):
        """将音频张量转为WAV格式的二进制流"""
        # 创建内存缓冲区（替代本地文件）
        buffer = io.BytesIO()
        # 用ComfyUI的save_audio工具，将音频写入缓冲区
        save_audio(audio_tensor, buffer, sample_rate, format="wav")
        # 重置缓冲区指针到开头（否则上传会是空数据）
        buffer.seek(0)
        return buffer.read()

    def upload_to_qiniu(self, audio_bytes, filename):
        """用二进制流直接上传七牛云（无需本地文件路径）"""
        try:
            # from qiniu import Auth, put_data
            #
            # # 七牛云配置（替换为你的信息）
            # access_key = "你的AccessKey"
            # secret_key = "你的SecretKey"
            # bucket_name = "你的存储空间名称"
            # domain = "你的七牛云域名"  # 如：xxx.clouddn.com
            #
            # # 初始化认证
            # q = Auth(access_key, secret_key)
            # # 生成上传令牌（文件名用我们自定义的）
            # token = q.upload_token(bucket_name, filename, 3600)
            #
            # # 关键：用put_data直接上传二进制数据（无需本地文件）
            # ret, info = put_data(token, filename, audio_bytes)

            ret = None
            if ret is not None:
                audio_url = f"https://{domain}/{ret['key']}"
                print(f"音频已上传到七牛云：{audio_url}")
            else:
                print(f"七牛云上传失败：{info}")
        except Exception as e:
            print(f"七牛云上传出错：{str(e)}")