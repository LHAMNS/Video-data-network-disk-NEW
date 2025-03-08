"""
Web UI服务器，使用Flask和Socket.IO提供实时通信
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO
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

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from converter import (
    CacheManager, ReedSolomonEncoder, FrameGenerator,
    VideoEncoder, StreamingVideoEncoder, calculate_video_params
)
from converter.utils import is_nvenc_available, is_qsv_available

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化Flask和SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 项目根目录
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = ROOT_DIR / "cache"
OUTPUT_DIR = ROOT_DIR / "output"

# 确保目录存在
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 初始化缓存管理器
cache_manager = CacheManager(CACHE_DIR)

# 全局变量
current_task = None
conversion_active = False
task_progress = {
    "file_id": None,
    "status": "idle",
    "processed_frames": 0,
    "total_frames": 0,
    "fps": 0,
    "eta": 0,
    "preview_image": None,
    "start_time": 0,
    "output_path": None
}


class ConversionTask:
    """转换任务，管理文件到视频的转换过程"""
    
    def __init__(self, file_id, params):
        """
        初始化转换任务
        
        Args:
            file_id: 缓存的文件ID
            params: 转换参数字典
        """
        self.file_id = file_id
        self.params = params
        self.running = False
        self.thread = None
        self.frame_generator = None
        self.video_encoder = None
        self.start_time = 0
        self.processed_frames = 0
        self.total_frames = 0
        self.error_correction = None
        
        # 解析参数
        self.resolution = params.get("resolution", "4K")
        self.fps = int(params.get("fps", 30))
        self.nine_to_one = params.get("nine_to_one", True)
        self.color_count = int(params.get("color_count", 16))
        self.error_correction_enabled = params.get("error_correction", True)
        self.error_correction_ratio = float(params.get("error_correction_ratio", 0.1))
        self.quality = params.get("quality", "high")
        
        # 获取文件信息
        self.file_info = cache_manager.get_file_info(file_id)
        if not self.file_info:
            raise ValueError(f"找不到文件ID: {file_id}")
        
        self.file_size = self.file_info["file_size"]
        self.output_path = OUTPUT_DIR / f"{self.file_info['original_filename']}.mp4"
        
        # 计算视频参数
        self.video_params = calculate_video_params(
            self.file_size, self.resolution, self.fps, self.nine_to_one,
            self.color_count, self.error_correction_ratio
        )
        
        self.total_frames = self.video_params["total_frames"]
        logger.info(f"创建转换任务: {self.file_info['original_filename']}, "
                 f"{self.resolution}, {self.fps}fps, 9合1={self.nine_to_one}, "
                 f"预计{self.total_frames}帧")
    
    def start(self):
        """启动转换任务"""
        if self.running:
            logger.warning("任务已在运行")
            return
        
        self.running = True
        self.start_time = time.time()
        
        # 创建并启动工作线程
        self.thread = threading.Thread(target=self._conversion_worker)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"转换任务已启动: {self.file_info['original_filename']}")
    
    def stop(self):
        """停止转换任务"""
        if not self.running:
            return
        
        logger.info("停止转换任务...")
        self.running = False
        
        # 等待线程结束
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
        
        # 停止视频编码器
        if self.video_encoder:
            self.video_encoder.stop()
        
        logger.info("转换任务已停止")
    
    def _frame_generated_callback(self, frame_idx, total_frames, frame):
        """
        帧生成回调
        
        Args:
            frame_idx: 当前帧索引
            total_frames: 总帧数
            frame: 生成的帧
        """
        self.processed_frames = frame_idx + 1
        
        # 每10帧更新一次预览和进度
        if frame_idx % 10 == 0 or frame_idx == 0:
            # 生成预览图
            preview_image = self.frame_generator.generate_preview_image(frame)
            
            # 计算进度和ETA
            elapsed = time.time() - self.start_time
            fps = self.processed_frames / elapsed if elapsed > 0 else 0
            eta = (self.total_frames - self.processed_frames) / fps if fps > 0 else 0
            
            # 更新任务进度
            global task_progress
            task_progress.update({
                "status": "converting",
                "processed_frames": self.processed_frames,
                "total_frames": self.total_frames,
                "fps": fps,
                "eta": eta,
                "preview_image": preview_image
            })
            
            # 发送进度更新
            socketio.emit('progress_update', task_progress)
    
    def _conversion_worker(self):
        """转换工作线程"""
        try:
            # 初始化帧生成器
            self.frame_generator = FrameGenerator(
                resolution=self.resolution,
                fps=self.fps,
                color_count=self.color_count,
                nine_to_one=self.nine_to_one
            )
            
            # 计算物理分辨率
            physical_width = self.video_params["physical_width"]
            physical_height = self.video_params["physical_height"]
            
            # 初始化视频编码器
            self.video_encoder = StreamingVideoEncoder(
                width=physical_width,
                height=physical_height,
                fps=self.fps,
                output_path=self.output_path,
                quality=self.quality
            )
            
            # 启动视频编码器
            self.video_encoder.start()
            
            # 初始化纠错编码器（如果启用）
            if self.error_correction_enabled:
                redundancy_bytes = int(255 * self.error_correction_ratio)
                self.error_correction = ReedSolomonEncoder(redundancy_bytes=redundancy_bytes)
            
            # 读取缓存文件
            data_generator = cache_manager.read_cached_file(self.file_id)
            
            # 如果启用纠错，先对整个数据进行编码
            if self.error_correction_enabled:
                logger.info("应用纠错编码...")
                
                # 读取所有数据
                all_data = bytearray()
                for chunk in data_generator:
                    all_data.extend(chunk)
                
                # 应用纠错编码
                encoded_data = self.error_correction.encode_data(all_data)
                
                # 使用编码后的数据
                data_source = encoded_data
            else:
                # 直接使用原始数据
                data_source = data_generator
            
            # 生成帧并编码
            for frame in self.frame_generator.generate_frames_from_data(
                data_source, callback=self._frame_generated_callback
            ):
                if not self.running:
                    break
                
                # 添加帧到视频编码器
                self.video_encoder.add_frame(frame)
            
            # 完成编码
            if self.running:
                logger.info("所有帧已处理，等待编码完成...")
                self.video_encoder.stop()
                
                # 更新任务进度
                global task_progress
                task_progress.update({
                    "status": "completed",
                    "processed_frames": self.processed_frames,
                    "total_frames": self.total_frames,
                    "output_path": str(self.output_path)
                })
                
                # 发送完成通知
                socketio.emit('conversion_complete', {
                    "output_file": str(self.output_path),
                    "duration": time.time() - self.start_time
                })
        
        except Exception as e:
            logger.error(f"转换错误: {e}", exc_info=True)
            
            # 更新任务进度
            global task_progress
            task_progress.update({
                "status": "error",
                "error_message": str(e)
            })
            
            # 发送错误通知
            socketio.emit('conversion_error', {"error": str(e)})
        
        finally:
            # 确保标记任务为非运行状态
            self.running = False


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/hardware-info', methods=['GET'])
def hardware_info():
    """获取硬件信息"""
    return jsonify({
        "nvenc_available": is_nvenc_available(),
        "qsv_available": is_qsv_available()
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传文件处理"""
    if 'file' not in request.files:
        return jsonify({"error": "找不到文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            
            # 缓存文件
            file_id = cache_manager.cache_file(temp_file.name)
            
            # 获取文件信息
            file_info = cache_manager.get_file_info(file_id)
            
            # 删除临时文件
            os.unlink(temp_file.name)
        
        # 计算默认参数下的视频参数
        video_params = calculate_video_params(
            file_info["file_size"], 
            resolution="4K", 
            fps=30, 
            nine_to_one=True, 
            color_count=16, 
            error_correction_ratio=0.1
        )
        
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


@app.route('/api/start-conversion', methods=['POST'])
def start_conversion():
    """启动转换任务"""
    global current_task, conversion_active, task_progress
    
    # 检查是否有活跃的转换任务
    if conversion_active:
        return jsonify({"error": "已有转换任务在运行"}), 400
    
    try:
        data = request.json
        file_id = data.get("file_id")
        params = data.get("params", {})
        
        if not file_id:
            return jsonify({"error": "未指定文件ID"}), 400
        
        # 创建新的转换任务
        current_task = ConversionTask(file_id, params)
        
        # 重置进度
        task_progress = {
            "file_id": file_id,
            "status": "starting",
            "processed_frames": 0,
            "total_frames": current_task.total_frames,
            "fps": 0,
            "eta": 0,
            "preview_image": None,
            "start_time": time.time(),
            "output_path": None
        }
        
        # 标记转换为活跃状态
        conversion_active = True
        
        # 启动任务
        current_task.start()
        
        return jsonify({
            "success": True,
            "task_info": {
                "file_id": file_id,
                "total_frames": current_task.total_frames,
                "estimated_duration": current_task.video_params["duration_seconds"],
                "output_path": str(current_task.output_path)
            }
        })
    
    except Exception as e:
        logger.error(f"启动转换错误: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/stop-conversion', methods=['POST'])
def stop_conversion():
    """停止转换任务"""
    global current_task, conversion_active, task_progress
    
    if not conversion_active or current_task is None:
        return jsonify({"error": "没有活跃的转换任务"}), 400
    
    try:
        # 停止任务
        current_task.stop()
        
        # 更新状态
        conversion_active = False
        task_progress["status"] = "stopped"
        
        return jsonify({"success": True})
    
    except Exception as e:
        logger.error(f"停止转换错误: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/progress', methods=['GET'])
def get_progress():
    """获取当前进度"""
    global task_progress
    return jsonify(task_progress)


@app.route('/api/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """下载生成的文件"""
    try:
        file_path = OUTPUT_DIR / filename
        if not file_path.exists():
            return jsonify({"error": "文件不存在"}), 404
        
        return send_file(file_path, as_attachment=True)
    
    except Exception as e:
        logger.error(f"下载错误: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@socketio.on('connect')
def handle_connect():
    """处理客户端连接"""
    logger.info(f"客户端已连接: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    """处理客户端断开连接"""
    logger.info(f"客户端已断开: {request.sid}")


def run_server(host='127.0.0.1', port=8080, debug=False):
    """运行Web服务器"""
    logger.info(f"启动Web服务器: {host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    run_server(debug=True)
