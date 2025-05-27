"""
Web UI服务器，使用Flask和Socket.IO提供实时通信
Web UI server using Flask and Socket.IO for real-time communication
"""

from flask import Flask, render_template, request, jsonify, send_file, abort
from flask_socketio import SocketIO, emit
import os
import json
import time
import threading
import logging
from pathlib import Path
import base64
import sys
import tempfile
import shutil
import uuid
import subprocess
from datetime import datetime
import mimetypes
import hashlib
import numpy as np

# GPU模块条件导入 - 完整fallback机制
try:
    from converter.gpu_error_correction import get_optimal_error_corrector
    GPU_ERROR_CORRECTION_AVAILABLE = True
    logger.info("GPU error correction available")
except ImportError as e:
    logger.warning(f"GPU error correction not available: {e}")
    GPU_ERROR_CORRECTION_AVAILABLE = False

    def get_optimal_error_corrector(redundancy_ratio):
        from converter.error_correction import ReedSolomonEncoder
        redundancy_bytes = max(1, int(255 * redundancy_ratio))
        return ReedSolomonEncoder(redundancy_bytes=redundancy_bytes)

try:
    from converter.gpu_frame_generator import GPUFrameGenerator
    GPU_FRAME_GENERATION_AVAILABLE = True
    logger.info("GPU frame generation available")
except ImportError as e:
    logger.warning(f"GPU frame generation not available: {e}")
    GPU_FRAME_GENERATION_AVAILABLE = False

try:
    from converter.video_raptor_encoder import VideoRaptorEncoder
    RAPTOR_ENCODER_AVAILABLE = True
    logger.info("Raptor encoder available")
except ImportError as e:
    logger.warning(f"Raptor encoder not available: {e}")
    RAPTOR_ENCODER_AVAILABLE = False

    class VideoRaptorEncoder:
        def __init__(self, width, height, fps):
            self.width = width
            self.height = height
            self.fps = fps

        def create_metadata_frame(self, file_info):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def create_calibration_frame(self):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def _add_sync_pattern(self, frame):
            frame[::20, :] = 255
            frame[:, ::20] = 255

# 添加项目根目录到路径
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from converter import (
    CacheManager,
    ReedSolomonEncoder,
    FrameGenerator,
    OptimizedFrameGenerator,
    VideoEncoder,
    StreamingVideoEncoder,
    calculate_video_params,
)
from converter.utils import is_nvenc_available, is_qsv_available
from converter.pipeline import (
    ConversionPipeline,
    ErrorCorrectionStage,
    FrameGenerationStage,
    VideoEncodingStage,
)

# 配置日志
# Configure logging
log_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化Flask和SocketIO
# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16GB上传限制
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', ping_timeout=60, ping_interval=25)

# 项目根目录
# Project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = ROOT_DIR / "cache"
OUTPUT_DIR = ROOT_DIR / "output"
TEMP_DIR = ROOT_DIR / "temp"

# 确保目录存在
# Ensure directories exist
for directory in [CACHE_DIR, OUTPUT_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

# 初始化缓存管理器
# Initialize cache manager
cache_manager = CacheManager(CACHE_DIR)

# 全局变量 - 使用字典提供线程安全的访问
# Global variables - using dict for thread-safe access
conversion_tasks = {}
task_registry = {}

# 初始状态模板
# Initial state template
def create_task_progress(file_id):
    """创建任务进度记录"""
    return {
        "id": str(uuid.uuid4()),
        "file_id": file_id,
        "status": "idle",
        "processed_frames": 0,
        "total_frames": 0,
        "fps": 0,
        "eta": 0,
        "preview_image": None,
        "start_time": 0,
        "elapsed_time": 0,
        "output_path": None,
        "error_message": None,
        "last_update": time.time()
    }

# 任务锁，用于同步访问
task_lock = threading.RLock()

class ConversionTask:
    """转换任务，管理文件到视频的转换过程"""
    
    def __init__(self, file_id, params, task_id=None):
        """
        初始化转换任务
        
        Args:
            file_id: 缓存的文件ID
            params: 转换参数字典
            task_id: 任务ID (可选)
        """
        self.file_id = file_id
        self.params = params
        self.task_id = task_id or str(uuid.uuid4())
        self.running = False
        self.thread = None
        self.frame_generator = None
        self.video_encoder = None
        self.start_time = 0
        self.processed_frames = 0
        self.last_frame_time = 0
        self.last_progress_update = 0
        self.total_frames = 0
        self.error_correction = None
        self.output_file_verified = False
        self.event = threading.Event()  # 用于线程同步
        
        # 解析参数
        self.resolution = params.get("resolution", "4K")
        self.fps = int(params.get("fps", 30))
        self.nine_to_one = params.get("nine_to_one", True)
        self.color_count = int(params.get("color_count", 16))
        self.error_correction_enabled = params.get("error_correction", True)
        self.error_correction_ratio = float(params.get("error_correction_ratio", 0.1))
        self.quality = params.get("quality", "high")
        self.use_optimized_generator = params.get("use_optimized_generator", True)
        
        # 获取文件信息
        self.file_info = cache_manager.get_file_info(file_id)
        if not self.file_info:
            raise ValueError(f"找不到文件ID: {file_id}")
        
        self.file_size = self.file_info["file_size"]
        self.original_filename = self.file_info["original_filename"]
        self.output_path = OUTPUT_DIR / f"{self.original_filename}_{int(time.time())}.avi"
        
        # 计算视频参数
        self.video_params = calculate_video_params(
            self.file_size, self.resolution, self.fps, self.nine_to_one,
            self.color_count, self.error_correction_ratio
        )
        
        self.total_frames = self.video_params["total_frames"]
        logger.info(f"创建转换任务 [{self.task_id}]: {self.original_filename}, "
                  f"{self.resolution}, {self.fps}fps, 9合1={self.nine_to_one}, "
                  f"预计{self.total_frames}帧")
    
    def start(self):
        """启动转换任务"""
        if self.running:
            logger.warning(f"任务 [{self.task_id}] 已在运行")
            return False
        
        self.running = True
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        self.last_progress_update = self.start_time
        self.event.clear()
        
        # 创建并启动工作线程
        self.thread = threading.Thread(target=self._conversion_worker)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"转换任务 [{self.task_id}] 已启动: {self.original_filename}")
        return True
    
    def stop(self, timeout=30):
        """
        停止转换任务
        
        Args:
            timeout: 等待线程结束的超时时间(秒)
            
        Returns:
            bool: 是否成功停止
        """
        if not self.running:
            return True
        
        logger.info(f"停止转换任务 [{self.task_id}]...")
        self.running = False
        self.event.set()  # 通知线程停止
        
        success = True
        
        # 等待线程结束
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                logger.warning(f"任务 [{self.task_id}] 线程未在超时时间内结束")
                success = False
        
        # 停止视频编码器
        if self.video_encoder:
            try:
                logger.info(f"停止视频编码器 [{self.task_id}]")
                self.video_encoder.stop()
            except Exception as e:
                logger.error(f"停止视频编码器出错 [{self.task_id}]: {e}")
                success = False
        
        # 最终状态更新
        self._update_task_status("stopped")
        
        logger.info(f"转换任务 [{self.task_id}] 已停止, 成功: {success}")
        return success
    
    def _update_task_status(self, status, error_message=None, output_path=None):
        """更新任务状态"""
        with task_lock:
            if self.task_id not in task_registry:
                return
                
            progress = task_registry[self.task_id]
            progress["status"] = status
            progress["processed_frames"] = self.processed_frames
            progress["total_frames"] = self.total_frames
            progress["last_update"] = time.time()
            progress["elapsed_time"] = time.time() - self.start_time
            
            if error_message:
                progress["error_message"] = error_message
                
            if output_path:
                progress["output_path"] = str(output_path)
    def _frame_generated_callback(self, frame_idx, total_frames, frame):
        """
        帧生成回调
        
        Args:
            frame_idx: 当前帧索引
            total_frames: 总帧数
            frame: 生成的帧
        """
        if not self.running:
            return
                
        current_time = time.time()
        self.processed_frames = frame_idx + 1
        frame_interval = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # 根据处理速度自适应调整更新频率
        update_interval = max(0.5, min(5.0, frame_interval * 10))
        
        # 定期更新进度
        if (current_time - self.last_progress_update >= update_interval or 
            frame_idx == 0 or (total_frames is not None and frame_idx >= total_frames - 1)):
            
            self.last_progress_update = current_time
            
            try:
                # 生成预览图
                preview_image = self.frame_generator.generate_preview_image(frame)
                
                # 计算进度和ETA
                elapsed = current_time - self.start_time
                fps = self.processed_frames / elapsed if elapsed > 0 else 0
                
                # 修复：处理total_frames为None的情况
                if total_frames is not None:
                    eta = (total_frames - self.processed_frames) / fps if fps > 0 else 0
                elif self.total_frames is not None:
                    eta = (self.total_frames - self.processed_frames) / fps if fps > 0 else 0
                else:
                    eta = 0  # 如果总帧数未知，则设置默认ETA
                
                # 更新任务进度
                with task_lock:
                    if self.task_id in task_registry:
                        progress = task_registry[self.task_id]
                        progress.update({
                            "status": "converting",
                            "processed_frames": self.processed_frames,
                            "total_frames": self.total_frames,
                            "fps": fps,
                            "eta": eta,
                            "preview_image": preview_image,
                            "elapsed_time": elapsed,
                            "last_update": current_time
                        })
                
                # 发送进度更新
                socketio.emit('progress_update', task_registry[self.task_id])
                
            except Exception as e:
                logger.error(f"进度更新错误 [{self.task_id}]: {e}", exc_info=True)        
    def _verify_output_video(self):
        """验证输出AVI文件 - 针对未压缩AVI优化"""
        if not self.output_path.exists():
            return False, "输出文件不存在"

        if self.output_path.stat().st_size == 0:
            return False, "输出文件大小为0"

        try:
            from converter.avi_validator import validate_avi_file

            validation_results = validate_avi_file(
                str(self.output_path),
                callback=None
            )

            if validation_results['overall_success']:
                structure = validation_results['structure_validation']
                logger.info(
                    f"AVI验证成功: {structure['resolution']}, "
                    f"{structure['frame_count']} frames, {structure['fps']} fps"
                )
                return True, "AVI文件验证通过"
            else:
                error_details = []
                if 'structure_validation' in validation_results:
                    error_details.extend(validation_results['structure_validation'].get('error_details', []))
                if 'integrity_validation' in validation_results:
                    error_details.extend(validation_results['integrity_validation'].get('error_details', []))

                error_msg = "; ".join(error_details) if error_details else "未知验证错误"
                logger.error(f"AVI验证失败: {error_msg}")
                return False, f"AVI验证失败: {error_msg}"

        except ImportError:
            logger.warning("AVI validator not available, using basic validation")
            return True, "基础验证通过（验证器不可用）"
        except Exception as e:
            logger.error(f"AVI验证过程出错: {e}", exc_info=True)
            return False, f"验证过程出错: {str(e)}"
        
    def _conversion_worker(self):
        """GPU-aware conversion worker thread"""
        try:
            self._update_task_status("initializing")

            from converter.encoder import StreamingDirectAVIEncoder  # lazy import

            # ---------- 1. GPU error-correction ----------
            if self.error_correction_enabled and GPU_ERROR_CORRECTION_AVAILABLE:
                logger.info(f"Initializing GPU error correction [{self.task_id}]")
                self.error_correction = get_optimal_error_corrector(self.error_correction_ratio)
            elif self.error_correction_enabled:
                logger.info(f"Using CPU error correction [{self.task_id}]")
                self.error_correction = ReedSolomonEncoder(redundancy_bytes=int(255 * self.error_correction_ratio))

            # ---------- 2. Frame generator ----------
            try:
                if GPU_FRAME_GENERATION_AVAILABLE:
                    self.frame_generator = GPUFrameGenerator(
                        resolution=self.resolution,
                        fps=self.fps,
                        color_count=self.color_count,
                        nine_to_one=self.nine_to_one
                    )
                    logger.info(f"Using GPU-accelerated frame generator [{self.task_id}]")
                else:
                    raise RuntimeError("GPU not available")
            except (RuntimeError, Exception) as e:
                logger.info(f"GPU frame generation failed, using CPU: {e}")
                generator_class = OptimizedFrameGenerator if self.use_optimized_generator else FrameGenerator
                self.frame_generator = generator_class(
                    resolution=self.resolution,
                    fps=self.fps,
                    color_count=self.color_count,
                    nine_to_one=self.nine_to_one
                )
                logger.info(f"Using CPU frame generator [{self.task_id}]")

            # ---------- 3.  Encoder initialisation ----------
            physical_width  = self.video_params["physical_width"]
            physical_height = self.video_params["physical_height"]

            # Metadata encoder (Raptor) for calibration & sync
            self.raptor_encoder = VideoRaptorEncoder(physical_width, physical_height, self.fps)

            from converter.encoder import StreamingDirectAVIEncoder   # lazy import
            self.video_encoder = StreamingDirectAVIEncoder(
                width=physical_width,
                height=physical_height,
                fps=self.fps,
                output_path=self.output_path
            )
            self.video_encoder.start()

            # ---------- 4.  Metadata / calibration frames ----------
            if self.params.get("metadata_frames", True) and RAPTOR_ENCODER_AVAILABLE:
                logger.info(f"Adding metadata frames [{self.task_id}]")

                file_info = {
                    'filename': self.original_filename,
                    'size': self.file_size,
                    'checksum': hashlib.sha256(str(self.file_id).encode()).hexdigest()[:16],
                    'total_symbols': 0
                }

                self.video_encoder.add_frame(self.raptor_encoder.create_metadata_frame(file_info))
                self.video_encoder.add_frame(self.raptor_encoder.create_calibration_frame())

                sync_frame = np.zeros((physical_height, physical_width, 3), dtype=np.uint8)
                self.raptor_encoder._add_sync_pattern(sync_frame)
                self.video_encoder.add_frame(sync_frame)

                self.processed_frames = 3

            self._update_task_status("processing")

            # ---------- 5.  Read source data ----------
            data_generator = cache_manager.read_cached_file(self.file_id)
            all_data = bytearray()
            for chunk in data_generator:
                if not self.running:
                    logger.info(f"Data collection interrupted [{self.task_id}]")
                    self._update_task_status("stopped")
                    return
                all_data.extend(chunk)

            # ---------- 6.  Error-correction ----------
            if self.error_correction_enabled and self.error_correction:
                logger.info(f"Applying Raptor error correction [{self.task_id}]…")
                self._update_task_status("error_correction")
                try:
                    encoded_data, stats = self.error_correction.process_file_data(bytes(all_data))
                    logger.info(
                        f"Raptor encoding complete [{self.task_id}]: "
                        f"{stats['throughput_mbps']:.1f} MB/s, "
                        f"redundancy: {stats['redundancy_ratio']:.2%}"
                    )
                    data_source = encoded_data
                except Exception as e:
                    error_msg = f"Error correction failed: {e}"
                    logger.error(f"{error_msg} [{self.task_id}]", exc_info=True)
                    self._update_task_status("error", error_msg)
                    socketio.emit('conversion_error', {"error": error_msg, "task_id": self.task_id})
                    return
            else:
                data_source = bytes(all_data)

            self._update_task_status("converting")

            # ---------- 7.  Frame generation & encoding ----------
            for frame in self.frame_generator.generate_frames_from_data(
                    data_source, callback=self._frame_generated_callback):
                if not self.running:
                    logger.info(f"Frame generation interrupted [{self.task_id}]")
                    self._update_task_status("stopped")
                    return
                if not self.video_encoder.add_frame(frame):
                    error_msg = "Failed to add frame to video encoder"
                    logger.error(f"{error_msg} [{self.task_id}]")
                    self._update_task_status("error", error_msg)
                    socketio.emit('conversion_error', {"error": error_msg, "task_id": self.task_id})
                    return

            # ---------- 8.  Finalise ----------
            if self.running:
                logger.info(f"All frames processed [{self.task_id}], finalizing…")
                self._update_task_status("finalizing")
                try:
                    stats = self.video_encoder.stop()
                    logger.info(f"Encoding complete [{self.task_id}]: {stats}")

                    is_valid, err = self._verify_output_video()
                    self.output_file_verified = is_valid
                    if not is_valid:
                        logger.error(f"Output verification failed [{self.task_id}]: {err}")
                        self._update_task_status("error", f"Output verification failed: {err}")
                        socketio.emit('conversion_error',
                                      {"error": f"Output verification failed: {err}",
                                       "task_id": self.task_id})
                        return

                    self._update_task_status("completed", output_path=str(self.output_path))
                    socketio.emit('conversion_complete',
                                  {"output_file": str(self.output_path),
                                   "filename": self.output_path.name,
                                   "duration": time.time() - self.start_time,
                                   "frames": self.processed_frames,
                                   "task_id": self.task_id})
                except Exception as e:
                    error_msg = f"Encoding completion error: {e}"
                    logger.error(f"{error_msg} [{self.task_id}]", exc_info=True)
                    self._update_task_status("error", error_msg)
                    socketio.emit('conversion_error',
                                  {"error": error_msg, "task_id": self.task_id})

        except Exception as e:
            error_msg = f"Conversion error: {e}"
            logger.error(f"{error_msg} [{self.task_id}]", exc_info=True)
            self._update_task_status("error", error_msg)
            socketio.emit('conversion_error', {"error": error_msg, "task_id": self.task_id})

        finally:
            self.running = False
            if hasattr(self, 'video_encoder') and self.video_encoder:
                try:
                    if getattr(self.video_encoder, 'running', False):
                        self.video_encoder.stop()
                except Exception as e:
                    logger.error(f"Error stopping video encoder [{self.task_id}]: {e}")
            self.event.set()


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/hardware-info', methods=['GET'])
def hardware_info():
    """获取硬件信息"""
    try:
        nvenc = is_nvenc_available()
        qsv = is_qsv_available()
        
        # 获取系统信息
        system_info = {
            "cpu_count": os.cpu_count(),
            "python_version": sys.version.split()[0],
            "platform": sys.platform
        }
        
        # 获取FFmpeg版本
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                ffmpeg_version = result.stdout.split('\n')[0]
                system_info["ffmpeg_version"] = ffmpeg_version
        except Exception as e:
            logger.warning(f"获取FFmpeg版本时出错: {e}")
            system_info["ffmpeg_version"] = "未知"
        
        return jsonify({
            "nvenc_available": nvenc,
            "qsv_available": qsv,
            "system_info": system_info
        })
    except Exception as e:
        logger.error(f"获取硬件信息时出错: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传文件处理"""
    if 'file' not in request.files:
        return jsonify({"error": "找不到文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    
    # 创建临时文件
    temp_file_path = None
    try:
        # 创建唯一的临时文件
        temp_file_dir = TEMP_DIR / str(uuid.uuid4())
        temp_file_dir.mkdir(exist_ok=True)
        temp_file_path = temp_file_dir / file.filename
        
        # 保存上传的文件
        logger.info(f"保存上传文件到: {temp_file_path}")
        file.save(temp_file_path)
        
        # 缓存文件
        file_id = cache_manager.cache_file(temp_file_path)
        
        # 获取文件信息
        file_info = cache_manager.get_file_info(file_id)
        
        if not file_info:
            return jsonify({"error": "文件缓存失败"}), 500
        
        # 计算默认参数下的视频参数
        video_params = calculate_video_params(
            file_info["file_size"], 
            resolution="4K", 
            fps=30, 
            nine_to_one=True, 
            color_count=16, 
            error_correction_ratio=0.1
        )
        
        logger.info(f"文件上传成功: {file.filename}, ID: {file_id}, 大小: {file_info['file_size']/1024/1024:.2f} MB")
        
        return jsonify({
            "success": True,
            "file_id": file_id,
            "file_info": {
                "filename": file_info["original_filename"],
                "size": file_info["file_size"],
                "size_mb": file_info["file_size"] / (1024 * 1024)
            },
            "video_params": video_params
        })
    
    except Exception as e:
        logger.error(f"上传错误: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
    finally:
        # 清理临时文件
        try:
            if temp_file_path and temp_file_path.exists():
                temp_file_path.unlink()
            
            # 清理临时目录
            if temp_file_path:
                temp_dir = temp_file_path.parent
                if temp_dir.exists() and temp_dir.is_dir():
                    shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {e}")


@app.route('/api/start-conversion', methods=['POST'])
def start_conversion():
    """启动转换任务"""
    try:
        data = request.json
        file_id = data.get("file_id")
        params = data.get("params", {})
        
        if not file_id:
            return jsonify({"error": "未指定文件ID"}), 400
        
        # 创建任务ID
        task_id = str(uuid.uuid4())
        
        # 创建新的转换任务
        task = ConversionTask(file_id, params, task_id)
        
        # 初始化进度记录
        with task_lock:
            # 注册任务
            task_progress = create_task_progress(file_id)
            task_progress.update({
                "id": task_id,
                "status": "starting",
                "processed_frames": 0,
                "total_frames": task.total_frames,
                "fps": 0,
                "eta": 0,
                "preview_image": None,
                "start_time": time.time(),
                "output_path": None
            })
            
            # 存储任务和进度记录
            conversion_tasks[task_id] = task
            task_registry[task_id] = task_progress
        
        # 启动任务
        success = task.start()
        
        if not success:
            with task_lock:
                if task_id in conversion_tasks:
                    del conversion_tasks[task_id]
                if task_id in task_registry:
                    del task_registry[task_id]
            return jsonify({"error": "无法启动任务"}), 500
        
        return jsonify({
            "success": True,
            "task_id": task_id,
            "task_info": {
                "file_id": file_id,
                "total_frames": task.total_frames,
                "estimated_duration": task.video_params["duration_seconds"],
                "output_path": str(task.output_path)
            }
        })
    
    except Exception as e:
        logger.error(f"启动转换错误: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/stop-conversion', methods=['POST'])
def stop_conversion():
    """停止转换任务"""
    try:
        data = request.json
        task_id = data.get("task_id")
        
        if not task_id:
            return jsonify({"error": "未指定任务ID"}), 400
        
        # 检查任务是否存在
        with task_lock:
            if task_id not in conversion_tasks:
                return jsonify({"error": "找不到指定的任务"}), 404
            
            task = conversion_tasks[task_id]
        
        # 停止任务
        success = task.stop()
        
        # 更新状态
        with task_lock:
            if task_id in task_registry:
                task_registry[task_id]["status"] = "stopped"
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "message": "任务停止了，但可能没有完全清理"}), 200
    
    except Exception as e:
        logger.error(f"停止转换错误: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_progress(task_id):
    """获取指定任务的进度"""
    with task_lock:
        if task_id in task_registry:
            return jsonify(task_registry[task_id])
        return jsonify({"error": "找不到指定的任务"}), 404


@app.route('/api/tasks', methods=['GET'])
def get_all_tasks():
    """获取所有任务"""
    with task_lock:
        return jsonify(list(task_registry.values()))


@app.route('/api/download/<task_id>', methods=['GET'])
def download_file_by_task(task_id):
    """根据任务ID下载生成的文件"""
    try:
        with task_lock:
            if task_id not in task_registry:
                return jsonify({"error": "找不到指定的任务"}), 404
                
            task_info = task_registry[task_id]
            
            if task_info["status"] != "completed":
                return jsonify({"error": "任务尚未完成"}), 400
                
            if not task_info.get("output_path"):
                return jsonify({"error": "没有可用的输出文件"}), 404
            
            file_path = Path(task_info["output_path"])
        
        if not file_path.exists():
            return jsonify({"error": "文件不存在"}), 404
        
        # 获取原始文件名
        if task_id in conversion_tasks:
            original_filename = conversion_tasks[task_id].original_filename
            filename = f"{os.path.splitext(original_filename)[0]}.avi"
        else:
            filename = file_path.name
        
        # 获取MIME类型
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = 'video/x-msvideo'
        
        return send_file(
            file_path, 
            as_attachment=True,
            download_name=filename,
            mimetype=mime_type
        )
    
    except Exception as e:
        logger.error(f"下载错误: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/download/file/<path:filename>', methods=['GET'])
def download_file_by_name(filename):
    """通过文件名下载文件"""
    try:
        file_path = OUTPUT_DIR / filename
        if not file_path.exists():
            return jsonify({"error": "文件不存在"}), 404
        
        # 获取MIME类型
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = 'video/x-msvideo'
        
        return send_file(
            file_path, 
            as_attachment=True,
            mimetype=mime_type
        )
    
    except Exception as e:
        logger.error(f"下载错误: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/verify-video/<task_id>', methods=['POST'])
def verify_video(task_id):
    """验证生成的AVI视频文件"""
    try:
        with task_lock:
            if task_id not in task_registry:
                return jsonify({"error": "找不到指定的任务"}), 404

            task_info = task_registry[task_id]

            if task_info["status"] != "completed":
                return jsonify({"error": "任务尚未完成"}), 400

            if not task_info.get("output_path"):
                return jsonify({"error": "没有可用的输出文件"}), 404

            output_path = Path(task_info["output_path"])

        if not output_path.exists():
            return jsonify({"error": "输出文件不存在"}), 404

        from converter.avi_validator import validate_avi_file

        def progress_callback(percentage, frame_num, bytes_processed):
            socketio.emit('verification_progress', {
                'task_id': task_id,
                'percentage': percentage,
                'frame': frame_num,
                'bytes': bytes_processed
            })

        validation_results = validate_avi_file(
            str(output_path),
            callback=progress_callback
        )

        socketio.emit('verification_complete', {
            'task_id': task_id,
            'results': validation_results
        })

        return jsonify({
            "success": True,
            "validation_results": validation_results
        })

    except Exception as e:
        logger.error(f"视频验证错误: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# 添加缺失的清理缓存API
@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """清理缓存文件"""
    try:
        # 使用已有的清理临时文件功能
        cleanup_temp_files(max_age=0)  # 设置为0表示清理所有临时文件
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"清理缓存出错: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/clean-tasks', methods=['POST'])
def clean_completed_tasks():
    """清理已完成的任务"""
    try:
        data = request.json
        status_filter = data.get("status", ["completed", "error", "stopped"])
        
        if not isinstance(status_filter, list):
            status_filter = [status_filter]
        
        cleaned_tasks = []
        
        with task_lock:
            # 找出要清理的任务
            tasks_to_clean = [
                task_id for task_id, info in task_registry.items()
                if info["status"] in status_filter
            ]
            
            # 清理任务
            for task_id in tasks_to_clean:
                if task_id in conversion_tasks:
                    # 确保任务已停止
                    task = conversion_tasks[task_id]
                    if task.running:
                        task.stop()
                    del conversion_tasks[task_id]
                
                # 记录任务信息然后删除
                if task_id in task_registry:
                    cleaned_tasks.append(task_registry[task_id])
                    del task_registry[task_id]
        
        return jsonify({
            "success": True,
            "cleaned_tasks": len(cleaned_tasks),
            "tasks": cleaned_tasks
        })
    
    except Exception as e:
        logger.error(f"清理任务错误: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@socketio.on('connect')
def handle_connect():
    """处理客户端连接"""
    logger.info(f"客户端已连接: {request.sid}")
    
    # 发送当前所有任务状态
    with task_lock:
        for task_id, progress in task_registry.items():
            emit('task_update', progress)


@socketio.on('disconnect')
def handle_disconnect():
    """处理客户端断开连接"""
    logger.info(f"客户端已断开: {request.sid}")


@socketio.on('get_tasks')
def handle_get_tasks():
    """处理获取所有任务请求"""
    with task_lock:
        emit('all_tasks', list(task_registry.values()))


@socketio.on('get_task')
def handle_get_task(data):
    """处理获取单个任务请求"""
    task_id = data.get('task_id')
    if not task_id:
        emit('error', {"error": "未指定任务ID"})
        return
        
    with task_lock:
        if task_id in task_registry:
            emit('task_update', task_registry[task_id])
        else:
            emit('error', {"error": "找不到指定的任务"})


def cleanup_temp_files(max_age=3600, max_size=None):
    """
    清理临时文件
    
    Args:
        max_age: 最大文件保留时间(秒)
        max_size: 临时目录最大大小(字节)
    """
    try:
        # 清理临时目录
        temp_dirs = [TEMP_DIR, Path(tempfile.gettempdir())]
        current_time = time.time()
        deleted_count = 0
        deleted_size = 0
        
        for temp_dir in temp_dirs:
            if not temp_dir.exists() or not temp_dir.is_dir():
                continue
                
            # 计算当前大小
            current_size = 0
            if max_size:
                for path, dirs, files in os.walk(temp_dir):
                    for f in files:
                        fp = os.path.join(path, f)
                        current_size += os.path.getsize(fp)
            
            # 如果大小超过限制，清理最旧的文件
            need_size_cleanup = max_size and current_size > max_size
            
            for item in temp_dir.iterdir():
                try:
                    is_old = False
                    is_temp = False
                    
                    # 检查是否是临时文件
                    is_temp = (
                        item.name.startswith('tmp') or 
                        item.name.startswith('temp') or
                        item.name.startswith('pymp-') or
                        '.tmp' in item.name
                    )
                    
                    # 检查文件年龄
                    if item.is_file():
                        is_old = current_time - item.stat().st_mtime > max_age
                    elif item.is_dir() and TEMP_DIR in temp_dirs and item.parent == TEMP_DIR:
                        # 只清理我们自己的临时目录中的子目录
                        try:
                            # 检查目录是否为UUID格式(我们自己创建的)
                            uuid.UUID(item.name)
                            is_temp = True
                            # 检查目录中最新文件的修改时间
                            latest_time = current_time
                            for child in item.glob('**/*'):
                                if child.is_file():
                                    latest_time = min(latest_time, child.stat().st_mtime)
                            is_old = current_time - latest_time > max_age
                        except ValueError:
                            pass
                    
                    # 删除旧的临时文件/目录
                    if (is_old and is_temp) or (need_size_cleanup and is_temp):
                        size = 0
                        if item.is_file():
                            size = item.stat().st_size
                            item.unlink()
                        elif item.is_dir():
                            for child in item.glob('**/*'):
                                if child.is_file():
                                    size += child.stat().st_size
                            shutil.rmtree(item, ignore_errors=True)
                        
                        deleted_count += 1
                        deleted_size += size
                        
                        # 如果已经清理了足够多的空间，停止清理
                        if need_size_cleanup and current_size - deleted_size <= max_size:
                            need_size_cleanup = False
                
                except Exception as e:
                    logger.warning(f"清理临时文件 {item} 时出错: {e}")
        
        if deleted_count > 0:
            logger.info(f"清理了 {deleted_count} 个临时文件/目录, 释放了 {deleted_size/1024/1024:.2f} MB")
    
    except Exception as e:
        logger.error(f"清理临时文件时出错: {e}", exc_info=True)


def monitor_tasks():
    """监控任务状态，清理过期任务"""
    try:
        logger.debug("开始任务监控")
        
        with task_lock:
            current_time = time.time()
            expired_tasks = []
            
            # 检查超时的任务
            for task_id, info in task_registry.items():
                # 如果有活跃任务但进度停滞
                if info["status"] == "converting" and current_time - info["last_update"] > 300:  # 5分钟无更新
                    logger.warning(f"任务 {task_id} 可能已停止响应")
                    socketio.emit('task_warning', {
                        "task_id": task_id,
                        "message": "任务可能已停止响应",
                        "time_since_update": current_time - info["last_update"]
                    })
                
                # 清理非活跃的完成/错误任务(保留24小时)
                if info["status"] in ["completed", "error", "stopped"] and current_time - info["last_update"] > 86400:
                    expired_tasks.append(task_id)
            
            # 移除过期任务
            for task_id in expired_tasks:
                if task_id in conversion_tasks:
                    task = conversion_tasks[task_id]
                    if task.running:
                        task.stop()
                    del conversion_tasks[task_id]
                
                if task_id in task_registry:
                    logger.info(f"清理过期任务: {task_id}")
                    del task_registry[task_id]
    
    except Exception as e:
        logger.error(f"监控任务时出错: {e}", exc_info=True)


def start_background_tasks():
    """启动后台任务"""
    def task_monitor_worker():
        """任务监控工作线程"""
        while True:
            try:
                monitor_tasks()
                time.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger.error(f"任务监控线程错误: {e}", exc_info=True)
                time.sleep(300)  # 出错后等待5分钟再次尝试
    
    def temp_cleanup_worker():
        """临时文件清理工作线程"""
        while True:
            try:
                cleanup_temp_files(max_age=3600, max_size=10 * 1024 * 1024 * 1024)  # 10GB限制
                time.sleep(3600)  # 每小时清理一次
            except Exception as e:
                logger.error(f"文件清理线程错误: {e}", exc_info=True)
                time.sleep(3600 * 6)  # 出错后等待6小时再次尝试
    
    # 启动监控线程
    monitor_thread = threading.Thread(target=task_monitor_worker, daemon=True)
    monitor_thread.start()
    
    # 启动清理线程
    cleanup_thread = threading.Thread(target=temp_cleanup_worker, daemon=True)
    cleanup_thread.start()


def run_server(host='127.0.0.1', port=8080, debug=False):
    """
    运行Web服务器
    
    Args:
        host: 主机地址
        port: 端口号
        debug: 是否启用调试模式
    """
    logger.info(f"启动Web服务器: {host}:{port}")
    
    # 启动后台任务
    start_background_tasks()
    
    # 初始清理
    cleanup_temp_files()
    
    # 运行服务器
    try:
        socketio.run(
            app, 
            host=host, 
            port=port, 
            debug=debug, 
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在停止服务器...")
    except Exception as e:
        logger.error(f"服务器运行错误: {e}", exc_info=True)
    finally:
        # 停止所有活跃任务
        for task_id, task in conversion_tasks.items():
            if task.running:
                logger.info(f"停止任务 {task_id}")
                task.stop()


if __name__ == '__main__':
    run_server(debug=True)