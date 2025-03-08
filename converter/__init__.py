"""
文件到视频转换系统核心模块
性能优先，使用多种高级优化技术
"""

import os
import json
import logging
import time
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 常量定义
ROOT_DIR = Path(__file__).parent.parent
CACHE_DIR = ROOT_DIR / "cache"
OUTPUT_DIR = ROOT_DIR / "output"

# 确保目录存在
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 预定义的16色调色板 (精心设计以在压缩后保持区分度)
COLOR_PALETTE_16 = [
    (0, 0, 0),       # 黑
    (255, 255, 255), # 白
    (255, 0, 0),     # 红
    (0, 255, 0),     # 绿
    (0, 0, 255),     # 蓝
    (255, 255, 0),   # 黄
    (0, 255, 255),   # 青
    (255, 0, 255),   # 洋红
    (128, 0, 0),     # 暗红
    (0, 128, 0),     # 暗绿
    (0, 0, 128),     # 暗蓝
    (128, 128, 0),   # 橄榄
    (0, 128, 128),   # 暗青
    (128, 0, 128),   # 紫
    (128, 128, 128), # 灰
    (255, 128, 0),   # 橙
]

# 视频参数预设
VIDEO_PRESETS = {
    "4K": {
        "width": 3840,
        "height": 2160,
    },
    "1080p": {
        "width": 1920,
        "height": 1080,
    },
    "720p": {
        "width": 1280,
        "height": 720,
    }
}

# 预设FPS范围
MIN_FPS = 10
MAX_FPS = 60
DEFAULT_FPS = 30

# 缓存块大小 (以字节为单位)
CHUNK_SIZE = 1024 * 1024  # 1MB

from .utils import *
from .error_correction import ReedSolomonEncoder
from .frame_generator import FrameGenerator
from .encoder import VideoEncoder
