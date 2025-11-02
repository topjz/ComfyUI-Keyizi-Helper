from nodes.uploadToQiniu import UploadToQiniu

# 节点注册列表（ComfyUI会自动扫描并加载）
NODE_CLASS_MAPPINGS = {
    "UploadToQiniu": UploadToQiniu
}

# 节点显示名称（在UI中显示的名字）
NODE_DISPLAY_NAME_MAPPINGS = {
    "UploadToQiniu": "上传至七牛云"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
# 版本信息（可选）
__version__ = "0.1.0"