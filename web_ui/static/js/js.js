/**
 * 文件到视频转换器 - 前端JavaScript
 * 高性能实现，支持实时预览、Socket.IO通信和进度监控
 */

// ========== 全局变量 ==========
let socket;
let currentFileId = null;
let isConverting = false;
let downloadUrl = null;

// DOM元素引用
const elements = {
    // 文件导入
    fileInput: document.getElementById('file-input'),
    uploadBtn: document.getElementById('upload-btn'),
    dragDropArea: document.getElementById('drag-drop-area'),
    
    // 文件信息
    fileInfo: document.getElementById('file-info'),
    fileName: document.getElementById('file-name'),
    fileSize: document.getElementById('file-size'),
    estimatedDuration: document.getElementById('estimated-duration'),
    estimatedFrames: document.getElementById('estimated-frames'),
    estimatedSize: document.getElementById('estimated-size'),
    
    // 预览和进度
    previewContainer: document.getElementById('preview-container'),
    previewImage: document.getElementById('preview-image'),
    previewInfo: document.getElementById('preview-info'),
    progressBar: document.getElementById('progress-bar'),
    framesProcessed: document.getElementById('frames-processed'),
    processingSpeed: document.getElementById('processing-speed'),
    timeRemaining: document.getElementById('time-remaining'),
    
    // 设置
    resolution: document.getElementById('resolution'),
    fps: document.getElementById('fps'),
    fpsValue: document.getElementById('fps-value'),
    nineToOne: document.getElementById('nine-to-one'),
    errorCorrection: document.getElementById('error-correction'),
    errorCorrectionOptions: document.getElementById('error-correction-options'),
    errorCorrectionRatio: document.getElementById('error-correction-ratio'),
    errorRatioValue: document.getElementById('error-ratio-value'),
    quality: document.getElementById('quality'),
    
    // 按钮
    startBtn: document.getElementById('start-btn'),
    stopBtn: document.getElementById('stop-btn'),
    downloadBtn: document.getElementById('download-btn'),
    
    // 硬件信息
    hardwareInfo: document.getElementById('hardware-info'),
    
    // 模态框
    notificationModal: new bootstrap.Modal(document.getElementById('notification-modal')),
    modalTitle: document.getElementById('modal-title'),
    modalBody: document.getElementById('modal-body')
};

// ========== 初始化函数 ==========
function initApp() {
    // 初始化Socket.IO连接
    socket = io();
    
    // 设置Socket.IO事件处理
    setupSocketEvents();
    
    // 设置UI事件处理
    setupUIEvents();
    
    // 检查硬件加速可用性
    checkHardwareAcceleration();
    
    // 阻止浏览器默认的拖放行为
    preventDefaultDragDrop();
}

// ========== Socket.IO事件处理 ==========
function setupSocketEvents() {
    // 连接成功
    socket.on('connect', () => {
        console.log('Socket.IO连接成功');
    });
    
    // 连接错误
    socket.on('connect_error', (error) => {
        console.error('Socket.IO连接错误:', error);
        showNotification('错误', '无法连接到服务器，请刷新页面重试。');
    });
    
    // 进度更新
    socket.on('progress_update', (data) => {
        updateProgress(data);
    });
    
    // 转换完成
    socket.on('conversion_complete', (data) => {
        conversionComplete(data);
    });
    
    // 转换错误
    socket.on('conversion_error', (data) => {
        conversionError(data);
    });
}

// ========== UI事件处理 ==========
function setupUIEvents() {
    // 文件导入按钮点击
    elements.uploadBtn.addEventListener('click', () => {
        if (elements.fileInput.files.length > 0) {
            uploadFile(elements.fileInput.files[0]);
        } else {
            showNotification('提示', '请先选择文件');
        }
    });
    
    // 文件输入变化
    elements.fileInput.addEventListener('change', () => {
        if (elements.fileInput.files.length > 0) {
            elements.uploadBtn.textContent = '导入';
            elements.uploadBtn.classList.remove('btn-outline-secondary');
            elements.uploadBtn.classList.add('btn-primary');
        }
    });
    
    // 拖放文件区域事件
    setupDragDropEvents();
    
    // FPS滑块变化
    elements.fps.addEventListener('input', () => {
        elements.fpsValue.textContent = elements.fps.value;
        updateEstimates();
    });
    
    // 纠错比例滑块变化
    elements.errorCorrectionRatio.addEventListener('input', () => {
        const ratio = parseFloat(elements.errorCorrectionRatio.value);
        elements.errorRatioValue.textContent = `${Math.round(ratio * 100)}%`;
        updateEstimates();
    });
    
    // 纠错开关切换
    elements.errorCorrection.addEventListener('change', () => {
        elements.errorCorrectionOptions.style.display = elements.errorCorrection.checked ? 'block' : 'none';
        updateEstimates();
    });
    
    // 分辨率变化
    elements.resolution.addEventListener('change', updateEstimates);
    
    // 9合1开关切换
    elements.nineToOne.addEventListener('change', updateEstimates);
    
    // 开始转换按钮点击
    elements.startBtn.addEventListener('click', startConversion);
    
    // 停止转换按钮点击
    elements.stopBtn.addEventListener('click', stopConversion);
    
    // 下载按钮点击
    elements.downloadBtn.addEventListener('click', downloadVideo);
    
    // 其他设置变化事件
    elements.quality.addEventListener('change', updateEstimates);
}

// ========== 拖放文件处理 ==========
function setupDragDropEvents() {
    // 拖拽进入区域
    elements.dragDropArea.addEventListener('dragenter', (e) => {
        e.preventDefault();
        e.stopPropagation();
        elements.dragDropArea.classList.add('drag-active');
    });
    
    // 拖拽在区域内移动
    elements.dragDropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        elements.dragDropArea.classList.add('drag-active');
    });
    
    // 拖拽离开区域
    elements.dragDropArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        elements.dragDropArea.classList.remove('drag-active');
    });
    
    // 放下文件
    elements.dragDropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        elements.dragDropArea.classList.remove('drag-active');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            elements.fileInput.files = files;
            uploadFile(files[0]);
        }
    });
}

// 阻止浏览器默认的拖放行为
function preventDefaultDragDrop() {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.body.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        }, false);
    });
}

// ========== 文件上传处理 ==========
function uploadFile(file) {
    // 创建FormData对象
    const formData = new FormData();
    formData.append('file', file);
    
    // 显示上传中状态
    elements.uploadBtn.disabled = true;
    elements.uploadBtn.textContent = '上传中...';
    elements.fileInfo.classList.add('d-none');
    
    // 发送上传请求
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 上传成功
            currentFileId = data.file_id;
            displayFileInfo(data);
            enableStartButton();
        } else {
            // 上传失败
            showNotification('上传失败', data.error || '未知错误');
        }
    })
    .catch(error => {
        console.error('上传错误:', error);
        showNotification('上传错误', '网络错误或服务器异常');
    })
    .finally(() => {
        // 恢复上传按钮状态
        elements.uploadBtn.disabled = false;
        elements.uploadBtn.textContent = '导入';
    });
}

// 显示文件信息
function displayFileInfo(data) {
    const fileInfo = data.file_info;
    const videoParams = data.video_params;
    
    // 显示文件名和大小
    elements.fileName.textContent = fileInfo.filename;
    elements.fileSize.textContent = formatFileSize(fileInfo.size);
    
    // 显示预估视频信息
    elements.estimatedDuration.textContent = videoParams.duration_formatted;
    elements.estimatedFrames.textContent = formatNumber(videoParams.total_frames);
    elements.estimatedSize.textContent = formatFileSize(videoParams.estimated_video_size);
    
    // 显示文件信息区域
    elements.fileInfo.classList.remove('d-none');
    
    // 存储原始数据供后续估算使用
    elements.fileInfo.dataset.fileSize = fileInfo.size;
    
    // 更新预览区域提示
    elements.previewInfo.textContent = `文件已准备好：${fileInfo.filename}`;
}

// 更新估算
function updateEstimates() {
    if (!currentFileId || !elements.fileInfo.dataset.fileSize) return;
    
    // 获取当前设置
    const params = getConversionParams();
    const fileSize = parseInt(elements.fileInfo.dataset.fileSize);
    
    // 计算有效载荷比例（考虑纠错）
    let effectiveRatio = 1.0;
    if (params.error_correction) {
        effectiveRatio = 1.0 - params.error_correction_ratio;
    }
    
    // 计算每帧存储容量（不考虑纠错）
    let bytesPerFrame;
    const resolutionMultiplier = {
        '4K': 1.0,
        '1080p': 0.25,
        '720p': 0.11
    }[params.resolution] || 1.0;
    
    // 计算基础容量
    const baseCapacity = 8294400; // 4K下的基础容量 (3840*2160/8)，每像素4位
    bytesPerFrame = baseCapacity * resolutionMultiplier;
    
    // 应用9合1因素
    if (params.nine_to_one) {
        bytesPerFrame /= 9;
    }
    
    // 应用纠错比例
    bytesPerFrame *= effectiveRatio;
    
    // 计算帧数和时长
    const totalFrames = Math.ceil(fileSize / bytesPerFrame);
    const durationSeconds = totalFrames / params.fps;
    const durationFormatted = formatDuration(durationSeconds);
    
    // 估算视频大小 (假设码率为分辨率和帧率的函数)
    const width = params.resolution === '4K' ? 3840 : params.resolution === '1080p' ? 1920 : 1280;
    const height = params.resolution === '4K' ? 2160 : params.resolution === '1080p' ? 1080 : 720;
    
    // 根据质量设置估算码率
    const qualityMultiplier = {
        'high': 0.15,
        'medium': 0.1,
        'low': 0.05
    }[params.quality] || 0.1;
    
    const estimatedBitrate = width * height * params.fps * qualityMultiplier; // 每秒比特数
    const estimatedSize = estimatedBitrate * durationSeconds / 8; // 字节
    
    // 更新UI
    elements.estimatedDuration.textContent = durationFormatted;
    elements.estimatedFrames.textContent = formatNumber(totalFrames);
    elements.estimatedSize.textContent = formatFileSize(estimatedSize);
}

// 启用开始按钮
function enableStartButton() {
    elements.startBtn.disabled = false;
}

// ========== 转换操作 ==========
// 开始转换
function startConversion() {
    if (!currentFileId || isConverting) return;
    
    // 获取转换参数
    const params = getConversionParams();
    
    // 发送开始转换请求
    fetch('/api/start-conversion', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            file_id: currentFileId,
            params: params
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 转换开始
            isConverting = true;
            updateUIForConversionStarted();
            
            // 隐藏拖放区域
            elements.dragDropArea.classList.add('fade-out');
            
            // 显示初始进度
            resetProgress(data.task_info.total_frames);
        } else {
            // 开始失败
            showNotification('开始失败', data.error || '未知错误');
        }
    })
    .catch(error => {
        console.error('开始转换错误:', error);
        showNotification('错误', '无法启动转换任务');
    });
}

// 停止转换
function stopConversion() {
    if (!isConverting) return;
    
    // 发送停止转换请求
    fetch('/api/stop-conversion', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // 转换停止
            isConverting = false;
            updateUIForConversionStopped();
            showNotification('已停止', '转换任务已停止');
        } else {
            // 停止失败
            showNotification('停止失败', data.error || '未知错误');
        }
    })
    .catch(error => {
        console.error('停止转换错误:', error);
        showNotification('错误', '无法停止转换任务');
    });
}

// 下载视频
function downloadVideo() {
    if (!downloadUrl) return;
    
    // 创建下载链接
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = downloadUrl.split('/').pop();
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// 获取转换参数
function getConversionParams() {
    return {
        resolution: elements.resolution.value,
        fps: parseInt(elements.fps.value),
        nine_to_one: elements.nineToOne.checked,
        color_count: 16, // 固定为16色
        error_correction: elements.errorCorrection.checked,
        error_correction_ratio: parseFloat(elements.errorCorrectionRatio.value),
        quality: elements.quality.value
    };
}

// ========== 进度更新 ==========
// 重置进度
function resetProgress(totalFrames) {
    elements.progressBar.style.width = '0%';
    elements.progressBar.textContent = '0%';
    elements.framesProcessed.textContent = `0 / ${formatNumber(totalFrames)}`;
    elements.processingSpeed.textContent = '0 帧/秒';
    elements.timeRemaining.textContent = '计算中...';
    elements.previewImage.classList.add('d-none');
}

// 更新进度
function updateProgress(data) {
    // 如果状态不是转换中，则忽略
    if (data.status !== 'converting' && data.status !== 'completed') return;
    
    // 更新进度条
    const progressPercent = (data.processed_frames / data.total_frames * 100).toFixed(1);
    elements.progressBar.style.width = `${progressPercent}%`;
    elements.progressBar.textContent = `${progressPercent}%`;
    
    // 更新帧信息
    elements.framesProcessed.textContent = `${formatNumber(data.processed_frames)} / ${formatNumber(data.total_frames)}`;
    
    // 更新速度
    elements.processingSpeed.textContent = `${data.fps.toFixed(1)} 帧/秒`;
    
    // 更新剩余时间
    elements.timeRemaining.textContent = formatDuration(data.eta);
    
    // 更新预览图像
    if (data.preview_image) {
        elements.previewImage.src = `data:image/jpeg;base64,${data.preview_image}`;
        elements.previewImage.classList.remove('d-none');
    }
    
    // 如果完成了
    if (data.status === 'completed') {
        conversionComplete({
            output_file: data.output_path,
            duration: 0 // 不知道实际时长，但不重要
        });
    }
}

// 转换完成
function conversionComplete(data) {
    isConverting = false;
    
    // 设置下载URL
    downloadUrl = `/api/download/${data.output_file.split('/').pop()}`;
    
    // 更新UI
    updateUIForConversionCompleted();
    
    // 显示完成通知
    const duration = data.duration ? formatDuration(data.duration) : '';
    showNotification(
        '转换完成', 
        `视频已成功生成${duration ? '，用时' + duration : ''}。<br>点击"导出视频"按钮下载。`
    );
}

// 转换错误
function conversionError(data) {
    isConverting = false;
    
    // 更新UI
    updateUIForConversionStopped();
    
    // 显示错误通知
    showNotification('转换错误', data.error || '未知错误');
}

// ========== UI状态更新 ==========
// 转换开始时的UI更新
function updateUIForConversionStarted() {
    // 禁用设置
    elements.fileInput.disabled = true;
    elements.uploadBtn.disabled = true;
    elements.resolution.disabled = true;
    elements.fps.disabled = true;
    elements.nineToOne.disabled = true;
    elements.errorCorrection.disabled = true;
    elements.errorCorrectionRatio.disabled = true;
    elements.quality.disabled = true;
    
    // 更新按钮状态
    elements.startBtn.disabled = true;
    elements.stopBtn.disabled = false;
    elements.downloadBtn.disabled = true;
    
    // 更新预览区域状态
    elements.previewInfo.textContent = '转换中...';
}

// 转换停止时的UI更新
function updateUIForConversionStopped() {
    // 恢复设置
    elements.fileInput.disabled = false;
    elements.uploadBtn.disabled = false;
    elements.resolution.disabled = false;
    elements.fps.disabled = false;
    elements.nineToOne.disabled = false;
    elements.errorCorrection.disabled = false;
    elements.errorCorrectionRatio.disabled = false;
    elements.quality.disabled = false;
    
    // 更新按钮状态
    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
    elements.downloadBtn.disabled = true;
    
    // 更新预览区域状态
    elements.previewInfo.textContent = '已停止';
    
    // 恢复拖放区域
    elements.dragDropArea.classList.remove('fade-out');
}

// 转换完成时的UI更新
function updateUIForConversionCompleted() {
    // 恢复设置
    elements.fileInput.disabled = false;
    elements.uploadBtn.disabled = false;
    elements.resolution.disabled = false;
    elements.fps.disabled = false;
    elements.nineToOne.disabled = false;
    elements.errorCorrection.disabled = false;
    elements.errorCorrectionRatio.disabled = false;
    elements.quality.disabled = false;
    
    // 更新按钮状态
    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
    elements.downloadBtn.disabled = false;
    
    // 更新预览区域状态
    elements.previewInfo.textContent = '转换完成';
    
    // 设置下载按钮样式
    elements.downloadBtn.classList.add('pulse');
    setTimeout(() => {
        elements.downloadBtn.classList.remove('pulse');
    }, 6000);
}

// ========== 硬件加速检测 ==========
function checkHardwareAcceleration() {
    fetch('/api/hardware-info')
        .then(response => response.json())
        .then(data => {
            // 更新硬件信息区域
            let html = '<ul class="list-unstyled mb-0">';
            
            if (data.nvenc_available) {
                html += '<li class="hardware-available"><i class="bi bi-check-circle-fill"></i> NVIDIA NVENC: 可用</li>';
            } else {
                html += '<li class="hardware-unavailable"><i class="bi bi-x-circle"></i> NVIDIA NVENC: 不可用</li>';
            }
            
            if (data.qsv_available) {
                html += '<li class="hardware-available"><i class="bi bi-check-circle-fill"></i> Intel QuickSync: 可用</li>';
            } else {
                html += '<li class="hardware-unavailable"><i class="bi bi-x-circle"></i> Intel QuickSync: 不可用</li>';
            }
            
            if (!data.nvenc_available && !data.qsv_available) {
                html += '<li><small>将使用软件编码 (较慢)</small></li>';
            }
            
            html += '</ul>';
            
            elements.hardwareInfo.innerHTML = html;
        })
        .catch(error => {
            console.error('获取硬件信息错误:', error);
            elements.hardwareInfo.innerHTML = '<p class="text-danger">无法获取硬件加速信息</p>';
        });
}

// ========== 工具函数 ==========
// 显示通知
function showNotification(title, message) {
    elements.modalTitle.textContent = title;
    elements.modalBody.innerHTML = message;
    elements.notificationModal.show();
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 格式化时间
function formatDuration(seconds) {
    if (!seconds || isNaN(seconds) || seconds < 0) return '00:00:00';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

// 格式化数字（添加千位分隔符）
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// ========== 初始化应用 ==========
document.addEventListener('DOMContentLoaded', initApp);
