# api/index.py

import sys
from pathlib import Path

# 将项目根目录添加到 Python 模块搜索路径中
# 这样我们就可以导入 eztalk_proxy 包中的模块
# Vercel 通常会将项目的根目录作为当前工作目录
# 如果你的 eztalk_proxy 文件夹就在项目根目录下，
# 下面的路径调整可能不是严格必需的，但可以增加健壮性。

# 获取当前文件 (api/index.py) 所在的目录 (api)
current_dir = Path(__file__).resolve().parent
# 获取项目根目录 (api 目录的上一级)
project_root = current_dir.parent
# 将项目根目录添加到 sys.path，以便能找到 eztalk_proxy
sys.path.append(str(project_root))

# 从你的 eztalk_proxy 应用中导入 FastAPI app 实例
# 假设你的 FastAPI app 实例在 eztalk_proxy/main.py 中定义为 `app`
from eztalk_proxy.main import app

# Vercel 的 Python ASGI 运行时会自动寻找这个名为 `app` 的变量。
# 你不需要在这里运行 uvicorn。