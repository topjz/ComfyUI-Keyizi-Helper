import torch
import torchaudio
import os
import re
from comfy_api.latest import ComfyExtension, io

class UploadToOSS:
    # 节点类别（UI左侧菜单分类）
    CATEGORY = "Keyizi Helper"
    # 节点执行函数
    FUNCTION = "process_audio"

    # 输入参数定义
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),  # 接收TTS的音频数据对象
                "subtitle": ("STRING", {"default": ""}),  # 接收TTS的字幕
                "seed": ("INT", {"default": 0}),  # 接收TTS的seed
                "oss": (["Qiniu", "Aliyun", "Tencent", "Baidu"],), # OSS供应商
                "access_key": ("STRING", {"default": ""}),  # 七牛云AccessKey
                "secret_key": ("STRING", {"default": ""}),  # 七牛云SecretKey
                "bucket_name": ("STRING", {"default": ""}),  # 七牛云存储空间名称
                "domain": ("STRING", {"default": ""}),  # 七牛云域名（如：xxx.clouddn.com）
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": ""}),  # 可选前缀
                "audio_format": (["wav", "mp3"], {"default": "wav"}),  # 上传的音频格式
            }
        }

    # 输出类型定义
    RETURN_TYPES = ("STRING", "STRING")  # 输出张量
    RETURN_NAMES = ("fileName", "audio_url")  # 输出名称（UI中显示）

    # 处理音频
    def process_audio(self,
                      audio,
                      subtitle,
                      seed,
                      oss,
                      access_key,
                      secret_key,
                      bucket_name,
                      domain,
                      filename_prefix="",
                      audio_format="wav"):
        print(f"开始")
        # 2. 解析音频数据（核心兼容逻辑）
        audio_tensor, sample_rate = self.parse_audio_data(audio)

        # 3. 生成目标文件名
        filename = self.generate_filename(subtitle, filename_prefix, seed, audio_format)
        print(f"生成目标文件名：{filename}")

        # 4. 音频张量转二进制流（支持wav/mp3）
        audio_bytes = self.tensor_to_audio_bytes(audio_tensor, sample_rate, audio_format)
        print(f"音频转换完成（{audio_format}），大小：{len(audio_bytes) / 1024:.2f}KB")

        UPLOADER_HANDLES = {
            "Qiniu": upload_qiniu,
            "Aliyun": upload_aliyun,
            "Tencent": upload_aliyun,
            "Baidu": upload_baidu
        }

        # # 5. 上传至七牛云
        try:
            audio_url = UPLOADER_HANDLES.get(oss)(
                audio_bytes, filename, access_key, secret_key, bucket_name, domain
            )
            print(f"上传成功，URL：{audio_url}")
            return "11.wav", "https://www.domain.com/11.wav"
        except Exception as e:
            raise RuntimeError(f"上传出错：{str(e)}")

    # 解析音频数据
    def parse_audio_data(self, audio):
        """解析音频数据，兼容多种格式（waveform/samples/元组）"""
        try:
            # 适配上游节点：字段为 waveform（如Whisper相关节点）
            audio_tensor = audio["waveform"]
            sample_rate = audio["sample_rate"]
            print("已适配格式：audio['waveform'] + audio['sample_rate']")
        except KeyError:
            try:
                # 适配ComfyUI标准格式：字段为 samples
                audio_tensor = audio["samples"]
                sample_rate = audio["sample_rate"]
                print("已适配格式：audio['samples'] + audio['sample_rate']")
            except KeyError:
                # 适配旧版元组格式：(tensor, sample_rate)
                if isinstance(audio, tuple) and len(audio) >= 2:
                    audio_tensor, sample_rate = audio[0], audio[1]
                    print("已适配格式：(tensor, sample_rate) 元组")
                else:
                    # 所有格式都不匹配时报错
                    raise ValueError(
                        f"音频格式不支持！当前格式：{type(audio)}\n"
                        f"支持的格式：\n"
                        f"1. 字典 {'waveform': 张量, 'sample_rate': 整数}\n"
                        f"2. 字典 {'samples': 张量, 'sample_rate': 整数}\n"
                        f"3. 元组 (音频张量, 采样率)"
                    )

        # 验证解析结果
        if not isinstance(audio_tensor, torch.Tensor):
            raise TypeError(f"音频张量必须是torch.Tensor，实际是：{type(audio_tensor)}")
        if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
            raise ValueError(f"采样率必须是正整数，实际是：{sample_rate}（类型：{type(sample_rate)}）")

        # 确保采样率为整数
        return audio_tensor, int(sample_rate)

    # 生成文件名称
    def generate_filename(self, subtitle, prefix, seed, audio_format):
        """生成安全的七牛云存储文件名"""
        # 处理字幕（保留中英文和常见符号，替换特殊字符）
        if subtitle:
            safe_subtitle = re.sub(r'[^\w\u4e00-\u9fa5,.!? ]', '_', subtitle)
            safe_subtitle = safe_subtitle.strip().replace(' ', '_')[:20]  # 限制长度
            base_name = f"{safe_subtitle}_{seed}"
        else:
            base_name = f"{prefix}_{seed}" if prefix else f"tts_audio_{seed}"
        # 拼接格式后缀
        return f"{base_name}.{audio_format}"

    # tensor处理音频二进制流
    def tensor_to_audio_bytes(self, audio_tensor, sample_rate, audio_format):
        """将音频张量转换为二进制流（兼容1D/2D/3D张量）"""
        # 核心修复：处理3D张量（移除批次维度），统一转为2D格式
        if audio_tensor.ndim == 3:
            # 3D格式：(批次维度, 声道数, 采样点数) → 移除批次维度（取第0个批次）
            audio_tensor = audio_tensor.squeeze(0)  # 例：(1, 1, 44100) → (1, 44100)
            print(f"已将3D音频张量降维为2D：{audio_tensor.shape}")
        elif audio_tensor.ndim == 1:
            # 1D格式（单声道）→ 2D格式：(1, 采样点数)
            audio_tensor = audio_tensor.unsqueeze(0)
            print(f"已将1D音频张量升维为2D：{audio_tensor.shape}")
        elif audio_tensor.ndim == 2:
            # 已为2D格式（声道数×采样点数），无需处理
            print(f"音频张量为2D格式：{audio_tensor.shape}")
        else:
            raise ValueError(f"音频张量维度不支持！需为1D/2D/3D，实际为 {audio_tensor.ndim}D")

        # 使用标准库io模块，避免命名冲突
        import io
        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            src=audio_tensor,
            sample_rate=sample_rate,
            format=audio_format
        )
        buffer.seek(0)
        return buffer.read()

    # 上传七牛云
    def upload_to_qiniu(self, audio_bytes, filename, access_key, secret_key, bucket_name, domain):
        """用二进制流直接上传七牛云（无需本地文件路径）"""
        try:
            # from qiniu import Auth, put_data
            # q = Auth(access_key, secret_key)
            #
            # # 生成上传令牌（有效期3600秒）
            # token = q.upload_token(bucket_name, filename, 3600)
            #
            # # 上传二进制数据
            # ret, info = put_data(token, filename, audio_bytes)
            #
            # # 验证上传结果
            # if ret is None or info.status_code != 200:
            #     raise Exception(f"上传响应异常：{info}")
            return f"https://domain/ret"
            # return f"https://{domain}/{ret['key']}"
        except Exception as e:
            print(f"七牛云上传出错：{str(e)}")