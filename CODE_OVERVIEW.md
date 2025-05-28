# 关键函数和API关系概览

## main.py
- `main()` - 程序入口，解析命令行参数并调用 `run_server` 启动 Web 服务。
- `setup_enhanced_logging(level)` - 配置日志系统，同时记录 FFmpeg 日志。
- `check_dependencies()` - 检查 FFmpeg 及硬件加速（NVENC/QSV）。
- `check_environment()` - 创建并验证 `cache/` 与 `output/` 目录。

## converter 模块
- 全局常量定义于 `converter/__init__.py`，并导出核心类：
  - `FrameGenerator`/`OptimizedFrameGenerator`
  - `VideoEncoder`(别名 `DirectAVIEncoder`)、`StreamingVideoEncoder`、`BatchVideoEncoder`
  - `ReedSolomonEncoder`
- `utils.py` 提供:
  - `expand_pixels_9x1()`、`bytes_to_color_indices()`：Numba 加速的像素映射
  - `CacheManager`：文件分块缓存管理
  - `calculate_video_params()`：根据文件大小计算帧数等参数
  - `verify_video_file()`：利用 FFmpeg/ffprobe 检查输出视频
  - `is_nvenc_available()`、`is_qsv_available()`：检测硬件编码支持
- `frame_generator.py`:
  - `FrameGenerator.generate_frames_from_data()` 生成帧序列，可回调进度
  - `OptimizedFrameGenerator` 进一步利用预分配缓冲区提高性能
- `encoder.py`:
  - `DirectAVIEncoder`/`StreamingDirectAVIEncoder`/`ParallelDirectAVIEncoder`
  - `get_optimal_encoder()` 根据 CPU 核心数选择合适的编码器实现
- `decoder.py`:
  - `VideoDecoder.extract_data()` 从视频还原字节流
  - `ParallelVideoDecoder` 通过线程池并行解码
- `error_correction.py`:
  - `ReedSolomonEncoder`、`XORInterleaver`、`HybridErrorCorrection`
- `gpu_error_correction.py`:
  - `GPUReedSolomonEncoder` 使用 CUDA 进行 RS 编码
- `gpu_frame_generator.py`:
  - `GPUFrameGenerator` 在 GPU 上生成帧
- `avi_writer.py`:
  - `SimpleAVIWriter` 直接写入未压缩 AVI
  - `convert_bytes_to_avi()` 使用给定帧生成器批量写入
- `pipeline.py`:
  - `ConversionPipeline` 组织 `ErrorCorrectionStage`、`FrameGenerationStage`、`VideoEncodingStage`
- `video_raptor_encoder.py`:
  - `VideoRaptorEncoder` 生成包含元数据和校准图的帧

## web_ui/server.py
- Flask + Socket.IO Web 服务，关键路由：
  - `/api/upload` 上传文件并缓存
  - `/api/start-conversion` 创建并启动 `ConversionTask`
  - `/api/stop-conversion` 停止指定任务
  - `/api/download/<task_id>` 下载输出文件
  - `ConversionTask` 内部调用 `FrameGenerator`, `VideoEncoder`, `ReedSolomonEncoder` 等执行完整流程。

## 其他
- 测试位于 `tests/test_frame_generator.py`，验证帧生成逻辑。

