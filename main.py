#!/usr/bin/env python3
"""
文件到视频转换系统 - 主程序
提供高性能、易用的文件到视频转换功能，支持硬件加速
"""

import os
import sys
import argparse
import logging
import multiprocessing as mp
from pathlib import Path
import time

# 添加项目目录到路径
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# 导入web服务器
from web_ui.server import run_server
from converter.utils import is_nvenc_available, is_qsv_available


def setup_enhanced_logging(level=logging.INFO):
    """
    Set up enhanced logging with file handler and more detailed formatting
    """
    # Create logs directory if it doesn't exist
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    log_file = log_dir / f"app_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Set up console handler with the same format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []  # Remove any existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log system information
    logger = logging.getLogger(__name__)
    logger.info(f"System: {sys.platform}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Log file: {log_file}")
    
    return log_file


def check_dependencies():
    """检查依赖项是否安装"""
    import subprocess
    
    logger = logging.getLogger(__name__)
    
    # 检查ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logger.info("FFmpeg 已安装")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg 未安装或无法运行，请安装 FFmpeg 后再运行本程序")
        return False
    
    # 检查硬件加速支持
    nvenc = is_nvenc_available()
    qsv = is_qsv_available()
    
    if nvenc:
        logger.info("检测到 NVIDIA NVENC 硬件加速")
    else:
        logger.info("未检测到 NVIDIA NVENC 硬件加速")
    
    if qsv:
        logger.info("检测到 Intel QuickSync 硬件加速")
    else:
        logger.info("未检测到 Intel QuickSync 硬件加速")
    
    if not nvenc and not qsv:
        logger.warning("未检测到硬件加速，将使用软件编码（较慢）")
    
    return True


def check_environment():
    """检查运行环境"""
    logger = logging.getLogger(__name__)
    
    # 检查缓存目录
    cache_dir = BASE_DIR / "cache"
    if not cache_dir.exists():
        cache_dir.mkdir()
        logger.info(f"已创建缓存目录: {cache_dir}")
    
    # 检查输出目录
    output_dir = BASE_DIR / "output"
    if not output_dir.exists():
        output_dir.mkdir()
        logger.info(f"已创建输出目录: {output_dir}")
    
    # 检查权限
    if not os.access(str(cache_dir), os.W_OK):
        logger.error(f"无法写入缓存目录: {cache_dir}")
        return False
    
    if not os.access(str(output_dir), os.W_OK):
        logger.error(f"无法写入输出目录: {output_dir}")
        return False
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文件到视频转换系统")
    parser.add_argument(
        "--host", default="127.0.0.1", 
        help="Web服务器主机地址 (默认: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, 
        help="Web服务器端口 (默认: 8080)"
    )
    parser.add_argument(
        "--debug", action="store_true", 
        help="启用调试模式"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = setup_enhanced_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("文件到视频转换系统启动")
    
    # 检查依赖项
    if not check_dependencies():
        sys.exit(1)
    
    # 检查环境
    if not check_environment():
        sys.exit(1)
    
    # 设置进程起始方法
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # 可能已经设置
    
    # 启动Web服务器
    try:
        logger.info(f"启动Web服务器: http://{args.host}:{args.port}")
        print(f"\n文件到视频转换系统已启动!\n")
        print(f"请使用浏览器访问: http://{args.host}:{args.port}\n")
        
        run_server(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭服务器")
    except Exception as e:
        logger.error(f"服务器运行错误: {e}", exc_info=True)
    
    logger.info("文件到视频转换系统已关闭")

<<<<<<< HEAD

=======
# 在main.py中添加性能优化选项

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文件到视频转换系统")
    parser.add_argument(
        "--host", default="127.0.0.1", 
        help="Web服务器主机地址 (默认: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, 
        help="Web服务器端口 (默认: 8080)"
    )
    parser.add_argument(
        "--debug", action="store_true", 
        help="启用调试模式"
    )
    parser.add_argument(
        "--max-performance", action="store_true", 
        help="启用最大性能模式 (使用最多系统资源)"
    )
    parser.add_argument(
        "--threads", type=int, 
        help="指定最大工作线程数"
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = setup_enhanced_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("文件到视频转换系统启动")
    
    # 配置性能选项
    if args.max_performance:
        logger.info("启用最大性能模式")
        os.environ["CONVERTER_MAX_PERFORMANCE"] = "1"
    
    if args.threads:
        logger.info(f"设置最大工作线程数: {args.threads}")
        os.environ["CONVERTER_MAX_THREADS"] = str(args.threads)
    
    # 检查依赖项
    if not check_dependencies():
        sys.exit(1)
    
    # 检查环境
    if not check_environment():
        sys.exit(1)
    
    # 设置进程起始方法
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # 可能已经设置
    
    # 启动Web服务器
    try:
        logger.info(f"启动Web服务器: http://{args.host}:{args.port}")
        print(f"\n文件到视频转换系统已启动!\n")
        print(f"请使用浏览器访问: http://{args.host}:{args.port}\n")
        
        run_server(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭服务器")
    except Exception as e:
        logger.error(f"服务器运行错误: {e}", exc_info=True)
    
    logger.info("文件到视频转换系统已关闭")
>>>>>>> 21916ba (Initial commit)
if __name__ == "__main__":
    main()