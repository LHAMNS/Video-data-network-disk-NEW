
⸻



1 | 运行时硬件探测

TARGET=$(./detect_gpu.sh)     # cuda|hip|opencl|base
docker run --gpus all -v $PWD/input:/in -v $PWD/out:/out \
           ghcr.io/yourorg/app:${TARGET} \
           -i /in/bigfile.bin -o /out/video.avi -r 30 -w 3840 -h 2160

detect_gpu.sh 逻辑：先看 /dev/nvidiactl → cuda；再看 /dev/kfd → hip；再看 /dev/dri → opencl；否则 base。

⸻

2 | 执行流程（8 大阶段、25 动作）

阶段	在线程 / 硬件	动作（按时序）	说明
0 准备	主线程 (CPU)	0-1 探测 GPU → 加载相应内核0-2 创建 video.avi，写 RIFF+hdrl+空头0-3 分配 pinned/显存缓冲，初始化队列 Q_raw、Q_disk	头部只缺 dwTotalFrames、movi size
1 读取	Reader 线程	1-1 顺序 aio_read 256 MiB → pin_in	2 MiB 对齐
2 GPU 打包	GPU stream	2-1 H→D pin_in→d_in2-2 pack4+LUT 内核：4 bpp→BGR24（可选9×上采样）2-3 D→H d_out→pin_rgb (24 MB/帧)	单帧 < 0.3 ms
3 写帧块	DiskWriter 线程	3-1 avi_writer.add_rgb_frame(pin_rgb)3-2 AVI 写 00dc size payload；偶数补 0	24 MB 写 30 ms (NVMe)
4 统计	同 DiskWriter	4-1 帧计数 ++4-2 判断是否读下一块	帧大小固定，直接累加
5 循环	两线程 + GPU	重复阶段 1-4 直到文件 EOF	GPU 与 I/O 重叠
6 收尾	主线程	6-1 等待队列清空6-2 回填 RIFF size、movi size、dwTotalFrames	3 次 seek+write
7 验证	Validator	7-1 ffprobe 检查 FourCC=’DIB ’, pix_fmt=rgb247-2 若失败抛异常；成功则输出统计	确保视频网站可识别
8 完毕	—	返回 {frames, size, path}	上传/归档


⸻

3 | 资源预算

缓冲 / 结构	大小	常驻位置
pin_in	256 MiB	Host RAM（pinned）
d_in	256 MiB	GPU 显存
d_out	128 MiB	GPU 显存
pin_rgb	128 MiB	Host RAM（pinned）
总显存	384 MiB	≤ 1 GB 阈值
总 RAM	≈ 600 MiB	≤ 1 GB 阈值


⸻

4 | 接口参数（cli.py）

usage: video-pipe [-i INPUT] [-o OUTPUT] [-w WIDTH] [-h HEIGHT] [-r FPS] [--chunk 256]

  -i, --input     源文件路径
  -o, --output    目标 AVI 文件
  -w, --width     逻辑宽度（像素）
  -h, --height    逻辑高度
  -r, --fps       帧率
  --chunk         读取块大小 (MiB，默认 256)


⸻

5 | 平台差异与自动降级

发现硬件	做法
/dev/nvidiactl	编译 CUDA 内核，启用 cudaMemcpyAsync；若存在 NVENC，可选后端转码
/dev/kfd	使用 HIP 内核；封装仍写 AVI
/dev/dri 但无 NVIDIA/AMD	OpenCL 1.2 内核；写 AVI
均无	纯 CPU 路径：NumPy bit-pack + RGB LUT；写 AVI


⸻

6 | 输出文件规格（供 QA 对照）
	•	容器：RIFF/AVI
	•	avih.dwFlags：0x10 (HAS_INDEX)
	•	strh.fccHandler："DIB " (uncompressed RGB)
	•	strf.biBitCount：24
	•	帧块标签："00db"（RGB raw）或 "00dc"（容错）
	•	每帧大小：width × height × 3 字节（下对齐偶数）
	•	结尾无 idx1（可选）；视频网站不依赖。

⸻

7 | 为什么足够高效
	•	GPU 打包+LUT：> 4 GB/s，几乎零开销
	•	CPU 只负责写磁盘：24 MB/帧 → 4 K 30 fps ≈ 746 MB/s，NVMe ×4 仍余裕
	•	显存/RAM 双 ≤ 1 GB：满足你最初限制
	•	无压缩 AVI：任何平台直接上传，后台自行转码，不需你再封装

⸻
