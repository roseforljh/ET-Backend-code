import sys
import os
from pathlib import Path

# 确保项目根目录在Python路径中
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ.setdefault("LOG_LEVEL", "INFO")

try:
    from eztalk_proxy.main import app
    print(f"Successfully imported app from eztalk_proxy.main")
except ImportError as e:
    print(f"Failed to import app: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path}")
    raise

# 确保应用可以被导入
__all__ = ["app"]