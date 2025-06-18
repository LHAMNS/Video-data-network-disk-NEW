在对整个项目进行全面审查后，我发现了多个需要修复的关键问题。让我先创建一个关键函数和API关系的详细文档，然后提供修复方案。:
# 文件到视频转换系统 - 关键函数和API关系文档

## 核心架构概览

该系统采用模块化设计，主要分为以下几个核心组件：

1. **帧生成器 (Frame Generator)** - 将二进制数据转换为视频帧
2. **编码器 (Encoder)** - 将帧序列编码为视频文件
3. **纠错模块 (Error Correction)** - 提供数据冗余保护
4. **Web服务器 (Web Server)** - 提供用户界面和API接口
5. **任务管理器 (Task Manager)** - 管理转换任务的生命周期

## 关键类和函数关系图

```
main.py
  └── web_ui.server.run_server()
       ├── ConversionTask (任务管理核心)
       │    ├── FrameGenerator/OptimizedFrameGenerator/GPUFrameGenerator
       │    ├── ReedSolomonEncoder/GPUOptimizedEncoder
       │    └── StreamingDirectAVIEncoder/DirectAVIEncoder
       ├── CacheManager (文件缓存管理)
       └── Socket.IO (实时通信)
```

## 1. 帧生成器模块 (converter/frame_generator.py)

### 核心类：FrameGenerator

**关键属性：**
- `physical_width/height`: 物理分辨率（实际视频尺寸）
- `logical_width/height`: 逻辑分辨率（数据存储尺寸）
- `bytes_per_frame`: 每帧可存储的字节数
- `color_lut`: 颜色查找表

**关键方法：**

#### `generate_frame(data_chunk, frame_index=0) -> np.ndarray`
- **功能**: 将字节数据块转换为RGB帧
- **输入**: 二进制数据块(bytes)，帧索引(int)
- **输出**: RGB帧数组，形状为(height, width, 3)
- **调用链**: 
  - `bytes_to_color_indices()` → 将字节映射到颜色索引
  - `generate_frame_array()` → 根据索引生成帧数组
  - `expand_pixels_9x1()` → 9合1像素扩展（如果启用）

#### `generate_frames_from_data(data, callback=None) -> Iterator[np.ndarray]`
- **功能**: 从数据生成帧序列的生成器
- **关键逻辑**: 
  ```python
  total_frames = (len(data) + bytes_per_frame - 1) // bytes_per_frame
  for frame_idx in range(total_frames):
      chunk = data[start:end]
      frame = generate_frame(chunk, frame_idx)
      if callback: callback(frame_idx, total_frames, frame)
      yield frame
  ```

### 优化类：OptimizedFrameGenerator

- 使用预分配缓冲区减少内存分配
- `logical_frame_buffer` 和 `physical_frame_buffer` 重复使用
- 向量化操作提升性能

### GPU加速类：GPUFrameGenerator (converter/gpu_frame_generator.py)

**关键CUDA内核：**
- `unpack_4bit_to_indices`: 4位数据解包
- `indices_to_rgb_direct`: 索引直接映射到RGB
- `upsample_9x1_optimized`: 9倍上采样

## 2. 编码器模块 (converter/encoder.py)

### 核心类：DirectAVIEncoder

**关键方法：**

#### `start() -> bool`
- 初始化AVI写入器
- 创建输出文件
- 返回是否成功启动

#### `add_frame(frame) -> bool`
- **关键调用**: `self.avi_writer.add_rgb_frame(frame)`
- 测量写入时间并更新统计
- 每100帧记录性能日志

#### `stop() -> dict`
- 关闭AVI文件
- 计算并返回编码统计信息
- **重要**: 必须调用以确保文件正确关闭

### 流式编码器：StreamingDirectAVIEncoder

- 添加帧队列(`frame_queue`)实现异步写入
- 后台线程(`_writer_loop`)处理实际写入
- 避免主线程阻塞

## 3. 纠错模块 (converter/error_correction.py)

### 核心类：ReedSolomonEncoder

**关键参数：**
- `redundancy_bytes`: 每块的冗余字节数（最大255）
- `chunk_size`: 块大小（RS编码限制为255）
- `num_workers`: 并行处理线程数

**关键方法：**

#### `encode_data(data) -> bytes`
- 分块并行处理大数据
- **调用流程**:
  ```python
  if len(data) <= self.data_bytes:
      return self.rs_codec.encode(data)
  else:
      return self._parallel_encode(data)
  ```

### GPU加速纠错：GPUReedSolomonEncoder (converter/gpu_error_correction.py)

- 使用CuPy进行GPU计算
- 预计算Galois域查找表
- CUDA内核实现并行编码

## 4. Web服务器模块 (web_ui/server.py)

### 核心类：ConversionTask

**生命周期方法：**

#### `__init__(file_id, params, task_id=None)`
- 初始化任务参数
- 计算视频参数
- 创建输出路径

#### `start() -> bool`
- 启动后台转换线程
- 更新任务状态为"starting"
- **关键**: 创建`_conversion_worker`线程

#### `_conversion_worker()`
**执行流程：**
1. 初始化纠错编码器（如果启用）
2. 创建帧生成器（GPU优先，CPU备选）
3. 初始化视频编码器
4. 添加元数据帧（可选）
5. 读取缓存数据
6. 应用纠错编码
7. 生成帧并编码
8. 验证输出文件
9. 发送完成通知

#### `_frame_generated_callback(frame_idx, total_frames, frame)`
- 更新进度信息
- 生成预览图像
- 通过Socket.IO推送进度
- **关键计算**: 
  ```python
  fps = processed_frames / elapsed_time
  eta = (total_frames - processed_frames) / fps
  ```

### 关键API端点

#### POST `/api/upload`
- 接收文件上传
- 调用`cache_manager.cache_file()`缓存文件
- 返回`file_id`和预估参数

#### POST `/api/start-conversion`
```python
task = ConversionTask(file_id, params, task_id)
task_registry[task_id] = create_task_progress(file_id)
conversion_tasks[task_id] = task
task.start()
```

#### Socket.IO事件
- `progress_update`: 发送转换进度
- `conversion_complete`: 转换完成通知
- `conversion_error`: 错误通知

## 5. 缓存管理器 (converter/utils.py)

### 类：CacheManager

**关键方法：**

#### `cache_file(filepath) -> str`
- 计算文件哈希作为ID
- 分块存储大文件
- 更新元数据
- **重要**: 返回的`file_id`用于后续操作

#### `read_cached_file(cache_id) -> Iterator[bytes]`
- 生成器模式读取缓存块
- 避免一次性加载整个文件到内存

## 6. AVI写入器 (converter/avi_writer.py)

### 类：SimpleAVIWriter

**关键方法：**

#### `add_rgb_frame(frame)`
- 转换RGB到BGR（OpenCV格式）
- 写入帧数据
- 更新帧计数器
- **注意**: 必须按顺序调用

#### `close()`
- 填充文件头中的占位符
- 更新总帧数
- **关键**: 不调用会导致文件损坏

## 7. 关键工具函数 (converter/utils.py)

#### `calculate_video_params(file_size, resolution, fps, nine_to_one, color_count, error_correction_ratio)`
- 计算所需帧数
- 估算视频时长
- 预测输出文件大小
- **核心公式**:
  ```python
  bytes_per_frame = logical_width * logical_height * bits_per_pixel // 8
  total_frames = (file_size + bytes_per_frame - 1) // bytes_per_frame
  ```

#### `is_nvenc_available()` / `is_qsv_available()`
- 检测硬件加速支持
- 通过FFmpeg查询编码器

## 8. 关键数据流

### 文件上传到视频生成完整流程：

1. **文件上传**
   ```
   客户端 → /api/upload → cache_manager.cache_file() → file_id
   ```

2. **开始转换**
   ```
   客户端 → /api/start-conversion → ConversionTask创建 → start()
   ```

3. **数据处理链**
   ```
   CacheManager.read_cached_file() 
     → ReedSolomonEncoder.encode_data() [可选]
     → FrameGenerator.generate_frames_from_data()
     → StreamingDirectAVIEncoder.add_frame()
   ```

4. **进度更新**
   ```
   _frame_generated_callback() → Socket.IO → 客户端UI更新
   ```

5. **完成处理**
   ```
   _verify_output_video() → Socket.IO完成事件 → 下载链接激活
   ```

## 9. 线程模型

- **主线程**: Flask/Socket.IO处理HTTP请求
- **转换线程**: 每个ConversionTask一个线程
- **编码器线程**: StreamingDirectAVIEncoder的后台写入线程
- **监控线程**: 定期清理过期任务和临时文件

## 10. 错误处理关键点

1. **资源管理**
   - 所有文件句柄必须在finally块中关闭
   - 视频捕获对象(cv2.VideoCapture)必须释放
   - 线程必须正确终止

2. **状态同步**
   - 使用`task_lock`保护任务注册表访问
   - Socket.IO事件可能在任务删除后触发

3. **GPU降级**
   - GPU不可用时自动降级到CPU实现
   - CuPy导入失败时使用CPUFallbackEncoder

## 11. 性能关键参数

- `bytes_per_frame`: 决定视频长度
- `max_blocks`: GPU批处理大小
- `frame_queue.maxsize`: 流式编码缓冲区大小
- `num_workers`: 并行处理线程数

## 12. 安全考虑

- 文件上传大小限制：16GB (`MAX_CONTENT_LENGTH`)
- 文件ID使用MD5哈希，避免路径遍历
- 临时文件使用UUID命名
- 定期清理过期文件

## 关键注意事项

1. **AVI文件必须正确关闭**，否则文件头不完整
2. **GPU内存有限**，批处理大小需要动态调整
3. **Socket.IO连接可能断开**，需要处理重连
4. **大文件处理**使用流式/分块，避免内存溢出
5. **线程安全**：任务注册表访问必须加锁

# 文件到视频转换系统 - 关键问题修复方案

## 1. 严重问题修复

### 1.1 main.py - 重复函数定义

**问题**: `setup_enhanced_logging`函数定义了两次

**修复**:
```python
# 删除第二个setup_enhanced_logging定义（第113-150行）
# 只保留第一个定义（第20-54行）
```

### 1.2 converter/encoder.py - 资源泄露

**问题**: AVI文件句柄可能在异常时未关闭

**修复**:
```python
class DirectAVIEncoder:
    def stop(self):
        if not self.running:
            return None
            
        self.running = False
        
        try:
            if self.avi_writer:
                self.avi_writer.close()
                self.avi_writer = None  # 添加这行
            # ... 其余代码
        except Exception as e:
            logger.error(f"Error stopping encoder: {e}", exc_info=True)
            # 确保在异常情况下也尝试关闭
            if self.avi_writer:
                try:
                    self.avi_writer.close()
                except:
                    pass
                self.avi_writer = None
            return None
```

### 1.3 converter/decoder.py - VideoCapture资源泄露

**问题**: 多个VideoCapture实例可能导致资源耗尽

**修复**:
```python
class ParallelVideoDecoder(VideoDecoder):
    def _process_frame_batch(self, start_frame, end_frame, callback):
        cap = None
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            # ... 处理代码
        finally:
            # 确保释放资源
            if cap is not None:
                cap.release()
        
        return batch_data
```

### 1.4 web_ui/server.py - 任务状态竞态条件

**问题**: Socket.IO事件可能在任务已删除后发送

**修复**:
```python
def _frame_generated_callback(self, frame_idx, total_frames, frame):
    if not self.running:
        return
    
    # ... 省略中间代码 ...
    
    # 发送进度更新前检查任务是否仍存在
    with task_lock:
        if self.task_id in task_registry:
            socketio.emit('progress_update', task_registry[self.task_id])
        else:
            logger.warning(f"Task {self.task_id} no longer in registry")
```

### 1.5 converter/gpu_error_correction.py - GPU初始化失败处理

**问题**: CuPy导入失败时系统崩溃

**修复**:
```python
# 在文件顶部添加
HAS_CUDA = False
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    logger.warning("CuPy not available, GPU acceleration disabled")
    cp = None

class GPUReedSolomonEncoder:
    def __init__(self, redundancy_bytes: int = 32, block_size: int = 255):
        if not HAS_CUDA:
            raise RuntimeError("CUDA/CuPy not available, use CPU implementation instead")
        # ... 其余初始化代码
```

### 1.6 converter/avi_writer.py - 文件头计算错误

**问题**: AVI头部偏移量计算可能错误

**修复**:
```python
class SimpleAVIWriter:
    def _write_headers(self):
        # ... 前面代码 ...
        
        # 修正avih帧数偏移量计算
        # RIFF(12) + LIST(8) + 'hdrl'(4) + 'avih'(4) + size(4) + 16 fields * 4 = 32 + 16
        self._avih_frames_offset = 12 + 8 + 4 + 4 + 4 + 16
        
        # ... 其余代码 ...
```

## 2. 性能问题修复

### 2.1 大文件内存问题

**问题**: 整个文件加载到内存

**修复 - web_ui/server.py**:
```python
def _conversion_worker(self):
    # ... 前面代码 ...
    
    # 改为流式处理
    if self.file_size > 100 * 1024 * 1024:  # 100MB以上使用流式
        # 创建数据流生成器
        def data_stream():
            buffer = bytearray()
            for chunk in cache_manager.read_cached_file(self.file_id):
                if not self.running:
                    return
                buffer.extend(chunk)
                # 当缓冲区足够大时，yield一部分
                while len(buffer) >= self.frame_generator.bytes_per_frame:
                    yield bytes(buffer[:self.frame_generator.bytes_per_frame])
                    buffer = buffer[self.frame_generator.bytes_per_frame:]
            if buffer:  # 处理剩余数据
                yield bytes(buffer)
        
        # 使用流式处理
        for frame in self.frame_generator.generate_frames_from_data(data_stream(), self._frame_generated_callback):
            if not self.running:
                break
            self.video_encoder.add_frame(frame)
    else:
        # 小文件使用原有逻辑
        # ... 原有代码 ...
```

### 2.2 GPU内存优化

**问题**: GPU批处理大小固定

**修复 - converter/gpu_frame_generator.py**:
```python
class GPUFrameGenerator:
    def _init_gpu_buffers(self):
        # 动态计算批处理大小
        available_memory = cp.cuda.Device().mem_info[0]
        frame_size = self.physical_height * self.physical_width * 3
        
        # 使用可用内存的50%，最多30帧
        self.max_batch_frames = min(30, int(available_memory * 0.5 / frame_size))
        self.max_batch_frames = max(1, self.max_batch_frames)  # 至少1帧
        
        logger.info(f"GPU batch size: {self.max_batch_frames} frames")
        
        # ... 其余缓冲区分配 ...
```

## 3. 安全问题修复

### 3.1 文件上传验证

**问题**: 没有文件类型检查

**修复 - web_ui/server.py**:
```python
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "找不到文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    
    # 添加文件大小检查
    file.seek(0, 2)  # 移到文件末尾
    file_size = file.tell()
    file.seek(0)  # 重置到开头
    
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({"error": f"文件太大，最大支持 {app.config['MAX_CONTENT_LENGTH'] / (1024*1024*1024):.1f} GB"}), 400
    
    # 添加文件扩展名白名单（可选）
    allowed_extensions = {'.txt', '.pdf', '.doc', '.docx', '.zip', '.rar', '.7z', '.tar', '.gz'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext and file_ext not in allowed_extensions:
        logger.warning(f"Unusual file extension uploaded: {file_ext}")
    
    # ... 其余原有代码 ...
```

### 3.2 路径遍历防护

**问题**: 文件下载可能存在路径遍历

**修复 - web_ui/server.py**:
```python
@app.route('/api/download/file/<path:filename>', methods=['GET'])
def download_file_by_name(filename):
    try:
        # 防止路径遍历
        safe_filename = Path(filename).name  # 只取文件名部分
        file_path = OUTPUT_DIR / safe_filename
        
        # 确保文件在OUTPUT_DIR内
        if not file_path.resolve().is_relative_to(OUTPUT_DIR.resolve()):
            logger.error(f"Path traversal attempt: {filename}")
            abort(403)
        
        if not file_path.exists():
            return jsonify({"error": "文件不存在"}), 404
        
        # ... 其余代码 ...
```

## 4. 错误处理改进

### 4.1 converter/error_correction.py - 线程池异常

**修复**:
```python
class ReedSolomonEncoder:
    def _parallel_encode(self, data):
        # ... 前面代码 ...
        
        # 并行处理所有块
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            try:
                # 提交所有任务
                futures = [executor.submit(encode_chunk, i) for i in range(total_chunks)]
                
                # 等待所有任务完成，捕获异常
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Chunk encoding failed: {e}")
                        # 可选：重试或使用备用方法
                        raise
            except Exception as e:
                # 确保线程池关闭
                executor.shutdown(wait=False)
                raise
        
        return bytes(output_buffer)
```

### 4.2 converter/utils.py - 缓存文件操作

**修复**:
```python
class CacheManager:
    def cache_file(self, filepath):
        filepath = Path(filepath)
        file_hash = self._calculate_file_hash(filepath)
        
        # 检查是否已缓存
        if file_hash in self.metadata:
            logger.info(f"文件已缓存: {filepath.name}")
            return file_hash
        
        # 创建新缓存条目
        file_size = filepath.stat().st_size
        chunks = []
        
        try:
            with open(filepath, 'rb') as src_file:
                chunk_index = 0
                while True:
                    data = src_file.read(self.chunk_size)
                    if not data:
                        break
                    
                    chunk_hash = hashlib.md5(data).hexdigest()
                    chunk_path = self.cache_dir / f"{file_hash}_{chunk_index}.bin"
                    
                    try:
                        with open(chunk_path, 'wb') as chunk_file:
                            chunk_file.write(data)
                    except IOError as e:
                        # 清理已写入的块
                        for chunk in chunks:
                            try:
                                Path(self.cache_dir / chunk["path"]).unlink()
                            except:
                                pass
                        raise IOError(f"Failed to write cache chunk: {e}")
                    
                    chunks.append({
                        "index": chunk_index,
                        "path": str(chunk_path.relative_to(self.cache_dir)),
                        "size": len(data),
                        "hash": chunk_hash
                    })
                    
                    chunk_index += 1
        except Exception as e:
            logger.error(f"Cache file failed: {e}")
            # 清理部分写入的文件
            for chunk in chunks:
                try:
                    Path(self.cache_dir / chunk["path"]).unlink()
                except:
                    pass
            raise
        
        # ... 其余代码 ...
```

## 5. 并发问题修复

### 5.1 任务注册表访问

**问题**: 任务注册表的某些访问没有加锁

**修复 - web_ui/server.py**:
```python
# 在所有访问task_registry的地方确保使用锁
def _update_task_status(self, status, error_message=None, output_path=None):
    with task_lock:  # 这行已有
        if self.task_id not in task_registry:
            logger.warning(f"Task {self.task_id} not in registry during status update")
            return  # 添加这行，避免后续错误
        
        # ... 其余代码 ...
```

## 6. 配置集中化

**问题**: 配置参数分散在多个地方

**建议创建 config.py**:
```python
# config.py
import os
from pathlib import Path

class Config:
    # 路径配置
    BASE_DIR = Path(__file__).resolve().parent
    CACHE_DIR = BASE_DIR / "cache"
    OUTPUT_DIR = BASE_DIR / "output"
    TEMP_DIR = BASE_DIR / "temp"
    LOG_DIR = BASE_DIR / "logs"
    
    # 服务器配置
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24))
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024 * 1024  # 16GB
    
    # 视频配置
    DEFAULT_FPS = 30
    DEFAULT_RESOLUTION = "4K"
    DEFAULT_COLOR_COUNT = 16
    
    # 性能配置
    MAX_WORKERS = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    GPU_BATCH_SIZE = 30
    STREAM_THRESHOLD = 100 * 1024 * 1024  # 100MB
    
    # 安全配置
    ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.doc', '.docx', '.zip', '.rar', '.7z', '.tar', '.gz'}
    
    # 清理配置
    TEMP_FILE_MAX_AGE = 3600  # 1小时
    TASK_EXPIRY_TIME = 86400  # 24小时
```

## 7. 测试覆盖

**建议添加单元测试**:
```python
# tests/test_frame_generator.py
import unittest
import numpy as np
from converter.frame_generator import FrameGenerator

class TestFrameGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = FrameGenerator(resolution="720p", fps=30)
    
    def test_bytes_per_frame_calculation(self):
        # 720p, 16色, 9合1模式
        # 逻辑分辨率: 426x240
        # 每像素4位
        expected = (426 * 240 * 4) // 8
        self.assertEqual(self.generator.bytes_per_frame, expected)
    
    def test_frame_generation(self):
        test_data = b'\x00' * self.generator.bytes_per_frame
        frame = self.generator.generate_frame(test_data)
        self.assertEqual(frame.shape, (720, 1280, 3))
```

## 8. 日志改进

**添加结构化日志**:
```python
import json
import logging

class StructuredLogger:
    def __init__(self, logger):
        self.logger = logger
    
    def log_event(self, event_type, **kwargs):
        log_entry = {
            'event': event_type,
            'timestamp': time.time(),
            **kwargs
        }
        self.logger.info(json.dumps(log_entry))

# 使用示例
structured_logger = StructuredLogger(logger)
structured_logger.log_event('conversion_start', 
                          task_id=task_id, 
                          file_size=file_size,
                          resolution=resolution)
```

## 总结

这些修复涵盖了：
1. 资源泄露问题
2. 并发安全问题
3. 错误处理改进
4. 性能优化
5. 安全加固
6. 代码组织改进

建议按优先级逐步实施这些修复，先处理严重的资源泄露和安全问题，然后改进性能和代码组织。

# Core dependencies
numpy>=1.21.0,<1.24.0  # 限制版本以兼容numba
numba==0.59.1
opencv-python>=4.9.0.0
flask>=2.0.1
flask-socketio>=5.1.1
Pillow>=8.3.1
psutil>=5.9.0
watchdog>=2.1.6
tqdm>=4.62.3
python-magic>=0.4.24
reedsolo>=1.5.4  # 仍然需要，用于CPU实现

# GPU acceleration (optional but recommended)
# 根据CUDA版本选择其中一个：
# For CUDA 12.x:
cupy-cuda12x>=12.0.0
# For CUDA 11.x (注释上面，取消注释下面):
# cupy-cuda11x>=11.0.0

# Web server dependencies
python-socketio[client]>=5.7.0
eventlet>=0.33.0  # 添加以提升Socket.IO性能

# 开发和测试依赖（可选）
# pytest>=7.0.0
# pytest-cov>=4.0.0
# black>=22.0.0
# flake8>=5.0.0


#!/usr/bin/env python3
"""
快速修复脚本 - 自动应用关键修复
运行: python apply_critical_fixes.py
"""

import os
import re
import shutil
from pathlib import Path

def backup_file(filepath):
    """创建文件备份"""
    backup_path = f"{filepath}.backup"
    if not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
        print(f"备份已创建: {backup_path}")

def fix_main_py():
    """修复main.py中的重复函数定义"""
    print("\n修复 main.py...")
    filepath = "main.py"
    
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return
    
    backup_file(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到第二个setup_enhanced_logging定义并删除
    pattern = r'(# Add this to setup_enhanced_logging.*?def setup_enhanced_logging.*?return log_file)'
    content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ 删除了重复的setup_enhanced_logging函数")

def fix_encoder_resource_leak():
    """修复encoder.py中的资源泄露"""
    print("\n修复 converter/encoder.py...")
    filepath = "converter/encoder.py"
    
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return
    
    backup_file(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 在DirectAVIEncoder.stop()中添加资源清理
    in_stop_method = False
    modified = False
    
    for i, line in enumerate(lines):
        if 'def stop(self):' in line and 'DirectAVIEncoder' in ''.join(lines[max(0, i-20):i]):
            in_stop_method = True
        
        if in_stop_method and 'if self.avi_writer:' in line and i+1 < len(lines) and 'self.avi_writer.close()' in lines[i+1]:
            # 在close()后添加置空
            if 'self.avi_writer = None' not in lines[i+2]:
                lines.insert(i+2, '                self.avi_writer = None\n')
                modified = True
                break
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("✓ 修复了AVI writer资源泄露")
    else:
        print("✗ 未找到需要修复的位置")

def fix_gpu_import():
    """修复GPU模块的导入错误处理"""
    print("\n修复 converter/gpu_error_correction.py...")
    filepath = "converter/gpu_error_correction.py"
    
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return
    
    backup_file(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加安全的导入检查
    if 'HAS_CUDA = False' not in content:
        import_section = '''import numpy as np
import logging
from typing import Optional, Tuple
import os

# 安全的CUDA导入
HAS_CUDA = False
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    logger.warning("CuPy not available, GPU acceleration disabled")
    cp = None

logger = logging.getLogger(__name__)'''
        
        content = re.sub(r'import numpy as np.*?logger = logging\.getLogger\(__name__\)', 
                        import_section, content, flags=re.DOTALL)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✓ 添加了安全的CUDA导入检查")

def fix_server_task_lock():
    """修复server.py中的任务锁问题"""
    print("\n修复 web_ui/server.py...")
    filepath = "web_ui/server.py"
    
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        return
    
    backup_file(filepath)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 在_frame_generated_callback中添加任务存在性检查
    pattern = r'(socketio\.emit\(\'progress_update\', task_registry\[self\.task_id\]\))'
    replacement = '''with task_lock:
                        if self.task_id in task_registry:
                            socketio.emit('progress_update', task_registry[self.task_id])
                        else:
                            logger.warning(f"Task {self.task_id} no longer in registry")'''
    
    if 'if self.task_id in task_registry:' not in content:
        content = re.sub(pattern, replacement, content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✓ 添加了任务注册表检查")

def create_config_file():
    """创建集中化配置文件"""
    print("\n创建 config.py...")
    
    config_content = '''"""
集中化配置文件
"""
import os
from pathlib import Path

class Config:
    # 路径配置
    BASE_DIR = Path(__file__).resolve().parent
    CACHE_DIR = BASE_DIR / "cache"
    OUTPUT_DIR = BASE_DIR / "output"
    TEMP_DIR = BASE_DIR / "temp"
    LOG_DIR = BASE_DIR / "logs"
    
    # 服务器配置
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24))
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024 * 1024  # 16GB
    
    # 视频配置
    DEFAULT_FPS = 30
    DEFAULT_RESOLUTION = "4K"
    DEFAULT_COLOR_COUNT = 16
    DEFAULT_NINE_TO_ONE = True
    
    # 性能配置
    MAX_WORKERS = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
    GPU_BATCH_SIZE = 30
    STREAM_THRESHOLD = 100 * 1024 * 1024  # 100MB
    FRAME_QUEUE_SIZE = 30
    
    # 纠错配置
    DEFAULT_REDUNDANCY_RATIO = 0.1
    RS_CHUNK_SIZE = 255
    
    # 安全配置
    ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.doc', '.docx', '.zip', '.rar', '.7z', '.tar', '.gz'}
    UPLOAD_TIMEOUT = 300  # 5分钟
    
    # 清理配置
    TEMP_FILE_MAX_AGE = 3600  # 1小时
    TASK_EXPIRY_TIME = 86400  # 24小时
    CACHE_CLEANUP_INTERVAL = 3600  # 1小时
    
    # Socket.IO配置
    SOCKETIO_PING_TIMEOUT = 60
    SOCKETIO_PING_INTERVAL = 25
    
    @classmethod
    def ensure_directories(cls):
        """确保所有必要的目录存在"""
        for dir_attr in ['CACHE_DIR', 'OUTPUT_DIR', 'TEMP_DIR', 'LOG_DIR']:
            directory = getattr(cls, dir_attr)
            directory.mkdir(exist_ok=True, parents=True)
'''
    
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("✓ 创建了集中化配置文件")

def fix_requirements():
    """修复requirements.txt"""
    print("\n修复 requirements.txt...")
    
    requirements_content = '''# Core dependencies
numpy>=1.21.0,<1.24.0  # 限制版本以兼容numba
numba==0.59.1
opencv-python>=4.9.0.0
flask>=2.0.1
flask-socketio>=5.1.1
Pillow>=8.3.1
psutil>=5.9.0
watchdog>=2.1.6
tqdm>=4.62.3
python-magic>=0.4.24
reedsolo>=1.5.4

# GPU acceleration (optional)
# For CUDA 12.x:
cupy-cuda12x>=12.0.0
# For CUDA 11.x (注释上面，取消注释下面):
# cupy-cuda11x>=11.0.0

# Web server dependencies
python-socketio[client]>=5.7.0
eventlet>=0.33.0
'''
    
    backup_file('requirements.txt')
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    
    print("✓ 修复了requirements.txt")

def add_error_handling_decorator():
    """创建错误处理装饰器"""
    print("\n创建 converter/decorators.py...")
    
    decorator_content = '''"""
错误处理和日志装饰器
"""
import functools
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

def safe_execute(default_return=None, log_errors=True):
    """
    安全执行装饰器，捕获异常并返回默认值
    
    Args:
        default_return: 异常时的默认返回值
        log_errors: 是否记录错误日志
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"{func.__name__} failed: {e}", exc_info=True)
                return default_return
        return wrapper
    return decorator

def with_timeout(timeout_seconds: float):
    """
    超时装饰器（仅用于同步函数）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"{func.__name__} timed out after {timeout_seconds}s")
            
            # 设置超时
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
            finally:
                # 恢复原处理器
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        return wrapper
    return decorator

def measure_performance(func: Callable) -> Callable:
    """
    性能测量装饰器
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper

def ensure_resource_cleanup(cleanup_func: Callable):
    """
    确保资源清理的装饰器
    
    Args:
        cleanup_func: 清理函数，接收self作为参数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            try:
                return func(self, *args, **kwargs)
            finally:
                try:
                    cleanup_func(self)
                except Exception as e:
                    logger.error(f"Cleanup failed in {func.__name__}: {e}")
        return wrapper
    return decorator
'''
    
    os.makedirs('converter', exist_ok=True)
    
    with open('converter/decorators.py', 'w', encoding='utf-8') as f:
        f.write(decorator_content)
    
    print("✓ 创建了错误处理装饰器")

def main():
    """运行所有修复"""
    print("开始应用关键修复...")
    print("=" * 50)
    
    fixes = [
        fix_main_py,
        fix_encoder_resource_leak,
        fix_gpu_import,
        fix_server_task_lock,
        create_config_file,
        fix_requirements,
        add_error_handling_decorator
    ]
    
    success_count = 0
    
    for fix_func in fixes:
        try:
            fix_func()
            success_count += 1
        except Exception as e:
            print(f"✗ 错误: {fix_func.__name__} - {e}")
    
    print("\n" + "=" * 50)
    print(f"完成！成功应用 {success_count}/{len(fixes)} 个修复")
    print("\n建议:")
    print("1. 检查所有 .backup 文件确保修复正确")
    print("2. 运行测试确保功能正常")
    print("3. 重新安装依赖: pip install -r requirements.txt")
    print("4. 重启服务器测试修复效果")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
关键功能单元测试
运行: python -m pytest test_critical_functions.py -v
"""

import unittest
import tempfile
import os
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestFrameGenerator(unittest.TestCase):
    """测试帧生成器"""
    
    def setUp(self):
        from converter.frame_generator import FrameGenerator
        self.generator = FrameGenerator(resolution="720p", fps=30, color_count=16, nine_to_one=True)
    
    def test_initialization(self):
        """测试初始化参数"""
        self.assertEqual(self.generator.fps, 30)
        self.assertEqual(self.generator.color_count, 16)
        self.assertEqual(self.generator.physical_width, 1280)
        self.assertEqual(self.generator.physical_height, 720)
        # 9合1模式下的逻辑尺寸
        self.assertEqual(self.generator.logical_width, 426)
        self.assertEqual(self.generator.logical_height, 240)
    
    def test_bytes_per_frame_calculation(self):
        """测试每帧字节数计算"""
        # 426 * 240 像素，每像素4位（16色）
        expected = (426 * 240 * 4) // 8
        self.assertEqual(self.generator.bytes_per_frame, expected)
    
    def test_frame_generation(self):
        """测试帧生成"""
        # 创建测试数据
        test_data = b'\x00' * self.generator.bytes_per_frame
        frame = self.generator.generate_frame(test_data, 0)
        
        # 验证输出尺寸
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(frame.dtype, np.uint8)
    
    def test_estimate_frame_count(self):
        """测试帧数估算"""
        file_size = 1024 * 1024  # 1MB
        expected = (file_size + self.generator.bytes_per_frame - 1) // self.generator.bytes_per_frame
        actual = self.generator.estimate_frame_count(file_size)
        self.assertEqual(actual, expected)

class TestAVIWriter(unittest.TestCase):
    """测试AVI写入器"""
    
    def setUp(self):
        from converter.avi_writer import SimpleAVIWriter
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "test.avi")
        self.writer = SimpleAVIWriter(640, 480, 30, self.output_path)
    
    def tearDown(self):
        """清理临时文件"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_write_single_frame(self):
        """测试写入单帧"""
        # 创建测试帧
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # 红色
        
        self.writer.open()
        self.writer.add_rgb_frame(frame)
        self.writer.close()
        
        # 验证文件存在且非空
        self.assertTrue(os.path.exists(self.output_path))
        self.assertGreater(os.path.getsize(self.output_path), 0)
    
    def test_multiple_frames(self):
        """测试写入多帧"""
        self.writer.open()
        
        # 写入10帧
        for i in range(10):
            frame = np.full((480, 640, 3), i * 25, dtype=np.uint8)
            self.writer.add_rgb_frame(frame)
        
        self.writer.close()
        
        # 验证帧数
        self.assertEqual(self.writer._frame_count, 10)

class TestCacheManager(unittest.TestCase):
    """测试缓存管理器"""
    
    def setUp(self):
        from converter.utils import CacheManager
        self.temp_dir = tempfile.mkdtemp()
        self.cache_manager = CacheManager(self.temp_dir)
        
        # 创建测试文件
        self.test_file = os.path.join(self.temp_dir, "test_file.bin")
        self.test_data = b"Hello World! " * 1000
        with open(self.test_file, 'wb') as f:
            f.write(self.test_data)
    
    def tearDown(self):
        """清理临时文件"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_cache_file(self):
        """测试文件缓存"""
        file_id = self.cache_manager.cache_file(self.test_file)
        
        # 验证返回了有效的文件ID
        self.assertIsNotNone(file_id)
        self.assertTrue(len(file_id) > 0)
        
        # 验证文件信息
        info = self.cache_manager.get_file_info(file_id)
        self.assertEqual(info['original_filename'], 'test_file.bin')
        self.assertEqual(info['file_size'], len(self.test_data))
    
    def test_read_cached_file(self):
        """测试读取缓存文件"""
        file_id = self.cache_manager.cache_file(self.test_file)
        
        # 读取缓存数据
        chunks = list(self.cache_manager.read_cached_file(file_id))
        reconstructed = b''.join(chunks)
        
        # 验证数据完整性
        self.assertEqual(reconstructed, self.test_data)

class TestErrorCorrection(unittest.TestCase):
    """测试纠错编码"""
    
    def setUp(self):
        from converter.error_correction import ReedSolomonEncoder
        self.encoder = ReedSolomonEncoder(redundancy_bytes=10)
    
    def test_encode_decode(self):
        """测试编码和解码"""
        # 测试数据
        original = b"Hello, World! This is a test message."
        
        # 编码
        encoded = self.encoder.encode_data(original)
        
        # 验证编码后数据更长
        self.assertGreater(len(encoded), len(original))
        
        # 解码
        decoded = self.encoder.decode_data(encoded, len(original))
        
        # 验证解码后数据匹配
        self.assertEqual(decoded, original)
    
    def test_large_data(self):
        """测试大数据编码"""
        # 创建1MB测试数据
        original = os.urandom(1024 * 1024)
        
        # 编码
        encoded = self.encoder.encode_data(original)
        
        # 解码
        decoded = self.encoder.decode_data(encoded, len(original))
        
        # 验证数据完整性
        self.assertEqual(decoded, original)

class TestVideoParams(unittest.TestCase):
    """测试视频参数计算"""
    
    def test_calculate_video_params(self):
        """测试视频参数计算"""
        from converter.utils import calculate_video_params
        
        file_size = 10 * 1024 * 1024  # 10MB
        params = calculate_video_params(
            file_size=file_size,
            resolution="1080p",
            fps=30,
            nine_to_one=True,
            color_count=16,
            error_correction_ratio=0.1
        )
        
        # 验证返回的参数
        self.assertIn('total_frames', params)
        self.assertIn('duration_seconds', params)
        self.assertIn('estimated_video_size', params)
        self.assertIn('logical_width', params)
        self.assertIn('logical_height', params)
        
        # 验证计算合理性
        self.assertGreater(params['total_frames'], 0)
        self.assertGreater(params['duration_seconds'], 0)
        self.assertEqual(params['logical_width'], 640)  # 1920/3
        self.assertEqual(params['logical_height'], 360)  # 1080/3

class TestResourceManagement(unittest.TestCase):
    """测试资源管理"""
    
    def test_file_handle_cleanup(self):
        """测试文件句柄清理"""
        from converter.avi_writer import SimpleAVIWriter
        
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        writer = SimpleAVIWriter(100, 100, 30, temp_path)
        writer.open()
        
        # 验证文件被打开
        self.assertIsNotNone(writer._f)
        
        writer.close()
        
        # 验证文件被关闭
        self.assertIsNone(writer._f)
        
        # 清理
        os.unlink(temp_path)

class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_end_to_end_conversion(self):
        """测试端到端转换流程"""
        from converter.frame_generator import FrameGenerator
        from converter.avi_writer import SimpleAVIWriter
        
        # 创建测试数据
        test_data = b"Hello World! " * 1000
        
        # 初始化组件
        generator = FrameGenerator(resolution="720p", fps=30)
        
        with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as f:
            output_path = f.name
        
        writer = SimpleAVIWriter(
            generator.physical_width,
            generator.physical_height,
            generator.fps,
            output_path
        )
        
        try:
            # 生成帧并写入
            writer.open()
            frame_count = 0
            
            for frame in generator.generate_frames_from_data(test_data):
                writer.add_rgb_frame(frame)
                frame_count += 1
            
            writer.close()
            
            # 验证
            self.assertGreater(frame_count, 0)
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)
            
        finally:
            # 清理
            if os.path.exists(output_path):
                os.unlink(output_path)

def run_tests():
    """运行所有测试"""
    unittest.main(verbosity=2)

if __name__ == '__main__':
    run_tests()

