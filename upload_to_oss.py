import torch
import torchaudio
import os
import re
import uuid
from comfy_api.latest import ComfyExtension, io
from qiniu import Auth, put_data

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
                "access_key": ("STRING", {"default": "jwPeIBNuakhyjYU5k6c9bS_NrYS2zaoD2eMQHKZx"}),  # 七牛云AccessKey
                "secret_key": ("STRING", {"default": "sTZ-0jgu3MR3Xo5MYAY3osHf7FBGQIASiQocYje4"}),  # 七牛云SecretKey
                "bucket_name": ("STRING", {"default": "byimg"}),  # 七牛云存储空间名称
                "domain": ("STRING", {"default": "https://img.bytide.net"}),  # 七牛云域名（如：xxx.clouddn.com）
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

        filename = f"{uuid.uuid4().hex}.{audio_format}"
        print(f"生成目标文件名：{filename}")

        # 4. 音频张量转二进制流（支持wav/mp3）
        audio_bytes = self.tensor_to_audio_bytes(audio_tensor, sample_rate, audio_format)
        print(f"音频转换完成（{audio_format}），大小：{len(audio_bytes) / 1024:.2f}KB")

        UPLOADER_HANDLES = {
            "Qiniu": self.upload_qiniu,
            "Aliyun": self.upload_aliyun,
            "Tencent": self.upload_tencent,
            "Baidu": self.upload_baidu
        }

        # # 5. 上传至七牛云
        try:
            audio_url = UPLOADER_HANDLES.get(oss)(audio_bytes, filename, access_key, secret_key, bucket_name, domain)
            print(f"上传成功，URL：{audio_url}")
            return filename, audio_url
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
        print(subtitle)
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
    def upload_qiniu(self, audio_bytes, filename, access_key, secret_key, bucket_name, domain):
        """用二进制流直接上传七牛云（无需本地文件路径）"""
        try:
            print(f"access_key：{access_key}，secret_key：{secret_key}")  # 打印部分令牌，确认生成正常
            # 初始化七牛云认证
            q = Auth(access_key, secret_key)

            # 生成上传令牌（有效期3600秒）
            token = q.upload_token(bucket_name, filename, 3600)
            print(f"七牛云上传令牌生成成功：{token[:30]}...")  # 打印部分令牌，确认生成正常

            # 上传二进制数据
            ret, info = put_data(token, filename, audio_bytes)
            print(f"七牛云上传响应：ret={ret}，info={info}")  # 打印完整响应，方便排查

            # 验证上传结果
            if ret is None:
                raise Exception(f"上传失败，七牛云未返回有效结果：{info}")
            if info.status_code not in [200, 201]:
                raise Exception(f"上传失败，HTTP状态码：{info.status_code}，详情：{info}")

            # 生成可访问的URL
            audio_url = f"{domain}/{ret['key']}"
            print(f"七牛云上传成功，URL：{audio_url}")
            return audio_url
        except ValueError as ve:
            # 配置格式错误，直接抛出
            raise RuntimeError(f"七牛云配置错误：{str(ve)}")
        except Exception as e:
            # 其他错误（认证失败、权限不足等）
            raise RuntimeError(
                f"七牛云上传出错：{str(e)}\n"
                f"排查建议：\n"
                f"1. 确认AK/SK填写正确（从七牛云密钥管理复制）\n"
                f"2. 确认bucket_name与控制台存储空间名称一致\n"
                f"3. 确认存储空间未被删除或禁用\n"
                f"4. 检查AK/SK是否有权限上传到该存储空间"
            )

    # 上传阿里云
    def upload_aliyun(self, audio_bytes, filename, access_key, secret_key, bucket_name, domain):
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
            return f"https://domain/aliyun"
            # return f"https://{domain}/{ret['key']}"
        except Exception as e:
            print(f"阿里云上传出错：{str(e)}")

    # 上传腾讯云
    def upload_tencent(self, audio_bytes, filename, access_key, secret_key, bucket_name, domain):
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
            return f"https://domain/tecent"
            # return f"https://{domain}/{ret['key']}"
        except Exception as e:
            print(f"腾讯云上传出错：{str(e)}")

    # 上传百度云
    def upload_baidu(self, audio_bytes, filename, access_key, secret_key, bucket_name, domain):
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
            return f"https://domain/baidu"
            # return f"https://{domain}/{ret['key']}"
        except Exception as e:
            print(f"百度云上传出错：{str(e)}")