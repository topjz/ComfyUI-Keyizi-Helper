from .upload_to_oss import UploadToOSS

NODE_CLASS_MAPPINGS = {
    "UploadToOSS": UploadToOSS
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UploadToOSS": "上传到OSS"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
# 版本信息（可选）
__version__ = "0.1.0"