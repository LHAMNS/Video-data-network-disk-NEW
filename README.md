# 文件到视频转换系统

该项目实现了一个将任意文件数据编码为视频并通过 Web 界面进行交互的完整流程，适用于需要在视频格式中嵌入或传输文件的场景。项目基于 Python 3，利用 Flask + Socket.IO 实现前端交互，底层使用 OpenCV、NumPy、FFmpeg 等工具完成帧生成与视频编码，并提供 Reed–Solomon 等纠错能力。

## 目录结构

- `main.py`：项目入口，负责环境检查、日志配置并启动 Web 服务。
- `run-script.bat`：在 Windows 环境下一键启动脚本。
- `requirements.txt`：运行所需的 Python 依赖列表。
- `converter/`：核心功能模块，包含文件到帧的转换、视频编码、纠错等实现。
  - `__init__.py`：全局常量与通用配置，如调色板、分辨率预设等。
  - `utils.py`：缓存管理、硬件加速检测及其他工具函数。
  - `frame_generator.py`：将字节数据映射为图像帧，可选择 9 合 1 像素模式以提高抗压缩能力。
  - `encoder.py`：调用 FFmpeg 对帧序列进行视频编码，支持流式与批量两种方式，并可使用 GPU 加速。
  - `decoder.py`：将生成的视频还原为原始数据，支持并行处理。
  - `avi_writer.py`：直接将 RGB 帧写入无压缩 AVI，速度仅受磁盘影响。
  - `error_correction.py`：实现 Reed–Solomon 及 XOR 交错等纠错算法。
- `web_ui/`：Web 前端及服务器逻辑。
  - `server.py`：Flask + Socket.IO 服务端，实现文件上传、任务管理、实时进度推送等功能。
  - `templates/index.html`：前端页面模板，提供文件上传、参数设置、进度显示等界面。
  - `static/js/js.js`：前端脚本，负责与 Socket.IO 通信并更新界面状态。
  - `static/css/css.css`：界面样式表。
- `cache/`、`output/`、`logs/`：分别存放缓存文件、生成的视频以及运行日志。

## 核心工作流程

1. **启动服务**：运行 `python main.py`（或在 Windows 使用 `run-script.bat`），系统会检查依赖与硬件加速能力，并启动 Web 服务器。
2. **文件上传**：在浏览器访问首页后，选择或拖放文件进行上传。文件首先被保存到临时目录并写入 `cache/`，同时计算视频参数（分辨率、帧数、预计大小等）。
3. **创建转换任务**：前端发送参数（分辨率、帧率、是否 9 合 1、纠错比例等）到 `/api/start-conversion`。`server.py` 创建 `ConversionTask` 并在后台线程中运行。
4. **帧生成**：任务读取缓存中的文件数据，通过 `FrameGenerator` 或 `OptimizedFrameGenerator` 逐块生成图像帧。如果启用纠错，则由 `ReedSolomonEncoder` 等模块在数据流中加入冗余信息。
5. **视频编码**：`StreamingVideoEncoder` 或 `BatchVideoEncoder` 将帧序列通过 FFmpeg 编码为 MP4 等格式，可根据硬件情况选择 NVENC/QSV/GPU 或软件编码方式。
6. **进度推送**：生成帧与编码过程中实时调用 Socket.IO 将当前帧数、预览图、估计剩余时间等信息推送至前端界面。
7. **任务完成**：编码结束后生成的视频文件保存在 `output/` 目录，并通过前端提供下载链接。系统还会验证视频文件完整性并记录日志。
8. **视频解码（可选）**：`decoder.py` 提供将视频恢复为原始数据的功能，便于校验或反向还原。

## 运行方式

1. 安装 Python 3.8 以上版本及 FFmpeg。
2. 执行 `pip install -r requirements.txt` 安装依赖。
3. 运行 `python main.py` 启动 Web 服务（默认监听 `127.0.0.1:8080`）。
4. 浏览器访问对应地址，上传文件并按需调整参数后开始转换。

Windows 用户可直接运行 `run-script.bat` 完成依赖安装和启动。

## 其他说明

- 所有日志文件位于 `logs/` 目录，便于排查问题。
- 转换过程中产生的中间数据会存放在 `cache/`，完成后的视频文件位于 `output/`。
- 若需自定义调色板、分辨率或编码器参数，可在 `converter` 模块中调整对应常量。

本仓库内的代码示例适合学习和实验用途，可根据需要进行扩展或集成。
