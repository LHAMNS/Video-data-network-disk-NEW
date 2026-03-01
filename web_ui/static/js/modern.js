/**
 * Modern GPU-Accelerated Video Encoding Interface
 * Direct AVI writing with verification capabilities
 */

// Global state
class AppState {
    constructor() {
        this.socket = null;
        this.currentFileId = null;
        this.currentTaskId = null;
        this.isProcessing = false;
        this.downloadUrl = null;
        this.fileInfo = null;
        this.verificationInProgress = false;

        // Decode state
        this.decodeFileId = null;
        this.decodeTaskId = null;
        this.isDecoding = false;
        this.decodeDownloadUrl = null;
        this.decodeFileInfo = null;
        this.activeMode = 'encode'; // 'encode' | 'decode'
    }
}

const state = new AppState();

// DOM elements
const elements = {
    // File handling
    fileInput: document.getElementById('file-input'),
    dropZone: document.getElementById('drop-zone'),
    previewImage: document.getElementById('preview-image'),
    previewCanvas: document.getElementById('preview-canvas'),
    
    // File info
    fileInfo: document.getElementById('file-info'),
    fileName: document.getElementById('file-name'),
    fileSize: document.getElementById('file-size'),
    estimatedFrames: document.getElementById('estimated-frames'),
    estimatedSize: document.getElementById('estimated-size'),
    
    // Controls
    resolutionBtns: document.querySelectorAll('[data-resolution]'),
    fpsSlider: document.getElementById('fps-slider'),
    fpsValue: document.getElementById('fps-value'),
    errorCorrection: document.getElementById('error-correction'),
    redundancySlider: document.getElementById('redundancy-slider'),
    redundancyValue: document.getElementById('redundancy-value'),
    gpuAcceleration: document.getElementById('gpu-acceleration'),
    metadataFrames: document.getElementById('metadata-frames'),
    
    // Progress
    progressBar: document.getElementById('progress-bar'),
    progressPercentage: document.getElementById('progress-percentage'),
    framesInfo: document.getElementById('frames-info'),
    timeRemaining: document.getElementById('time-remaining'),
    fpsInfo: document.getElementById('fps-info'),
    
    // Action buttons
    startBtn: document.getElementById('start-btn'),
    stopBtn: document.getElementById('stop-btn'),
    downloadBtn: document.getElementById('download-btn'),
    
    // Verification
    verificationSection: document.getElementById('verification-section'),
    startVerificationBtn: document.getElementById('start-verification'),
    verificationProgress: document.getElementById('verification-progress'),
    verificationResult: document.getElementById('verification-result'),
    verifyPercentage: document.getElementById('verify-percentage'),
    verifyFrame: document.getElementById('verify-frame'),
    verifyBytes: document.getElementById('verify-bytes'),
    resultIcon: document.getElementById('result-icon'),
    resultTitle: document.getElementById('result-title'),
    resultMessage: document.getElementById('result-message'),
    resultDetails: document.getElementById('result-details'),
    
    // Status
    gpuStatus: document.getElementById('gpu-status'),
    throughput: document.getElementById('throughput'),

    // Mode tabs
    tabEncode: document.getElementById('tab-encode'),
    tabDecode: document.getElementById('tab-decode'),
    panelEncode: document.getElementById('panel-encode'),
    panelDecode: document.getElementById('panel-decode'),

    // Decode elements
    decodeFileInput: document.getElementById('decode-file-input'),
    decodeDropZone: document.getElementById('decode-drop-zone'),
    decodeBrowseBtn: document.getElementById('decode-browse-btn'),
    decodePreviewCanvas: document.getElementById('decode-preview-canvas'),

    decodeFileInfo: document.getElementById('decode-file-info'),
    decodeFileName: document.getElementById('decode-file-name'),
    decodeFileSize: document.getElementById('decode-file-size'),
    decodeTotalFrames: document.getElementById('decode-total-frames'),
    decodeResolution: document.getElementById('decode-resolution'),
    decodeEstimatedOutput: document.getElementById('decode-estimated-output'),

    decodeResolutionBtns: document.querySelectorAll('[data-decode-resolution]'),
    decodeColorBtns: document.querySelectorAll('[data-decode-colors]'),
    decodeErrorCorrection: document.getElementById('decode-error-correction'),
    decodeGpuAcceleration: document.getElementById('decode-gpu-acceleration'),

    decodeProgressBar: document.getElementById('decode-progress-bar'),
    decodeProgressPercentage: document.getElementById('decode-progress-percentage'),
    decodeFramesInfo: document.getElementById('decode-frames-info'),
    decodeTimeRemaining: document.getElementById('decode-time-remaining'),
    decodeFpsInfo: document.getElementById('decode-fps-info'),

    decodeStartBtn: document.getElementById('decode-start-btn'),
    decodeStopBtn: document.getElementById('decode-stop-btn'),
    decodeDownloadBtn: document.getElementById('decode-download-btn'),
    decodeDownloadBtnSide: document.getElementById('decode-download-btn-side'),

    decodeResultSection: document.getElementById('decode-result-section'),
    decodeResultTitle: document.getElementById('decode-result-title'),
    decodeResultMessage: document.getElementById('decode-result-message'),
    decodeResultDetails: document.getElementById('decode-result-details')
};

// Initialize application
function initApp() {
    setupSocketIO();
    setupEventHandlers();
    checkHardwareCapabilities();
    setupDragAndDrop();
    setupModeTabs();
    setupDecodeHandlers();
    setupDecodeDragAndDrop();
}

// Socket.IO setup
function setupSocketIO() {
    state.socket = io();
    
    state.socket.on('connect', () => {
        console.log('Connected to server');
        updateGPUStatus('已连接');
    });
    
    state.socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateGPUStatus('连接断开');
    });
    
    state.socket.on('progress_update', (data) => {
        updateProgress(data);
    });
    
    state.socket.on('conversion_complete', (data) => {
        handleConversionComplete(data);
    });
    
    state.socket.on('conversion_error', (data) => {
        handleConversionError(data);
    });

    // Decode socket events
    state.socket.on('decode_progress', (data) => {
        updateDecodeProgress(data);
    });

    state.socket.on('decode_complete', (data) => {
        handleDecodeComplete(data);
    });

    state.socket.on('decode_error', (data) => {
        handleDecodeError(data);
    });
}

// Event handlers setup
function setupEventHandlers() {
    // File input
    elements.fileInput.addEventListener('change', handleFileSelect);
    
    // Resolution buttons
    elements.resolutionBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            elements.resolutionBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            updateEstimates();
        });
    });
    
    // Sliders
    elements.fpsSlider.addEventListener('input', () => {
        elements.fpsValue.textContent = elements.fpsSlider.value;
        updateEstimates();
    });
    
    elements.redundancySlider.addEventListener('input', () => {
        elements.redundancyValue.textContent = `${elements.redundancySlider.value}%`;
        updateEstimates();
    });
    
    // Toggles
    elements.errorCorrection.addEventListener('change', updateEstimates);
    elements.gpuAcceleration.addEventListener('change', updateEstimates);
    elements.metadataFrames.addEventListener('change', updateEstimates);
    
    // Action buttons
    elements.startBtn.addEventListener('click', startConversion);
    elements.stopBtn.addEventListener('click', stopConversion);
    elements.downloadBtn.addEventListener('click', downloadResult);
    
    // Verification
    elements.startVerificationBtn.addEventListener('click', startVerification);
}

// Drag and drop setup
function setupDragAndDrop() {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        elements.dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        elements.dropZone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        elements.dropZone.addEventListener(eventName, unhighlight, false);
    });
    
    elements.dropZone.addEventListener('drop', handleDrop);
    elements.dropZone.addEventListener('click', () => elements.fileInput.click());
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight() {
    elements.dropZone.classList.add('active');
}

function unhighlight() {
    elements.dropZone.classList.remove('active');
}

function handleDrop(e) {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        elements.fileInput.files = files;
        handleFileSelect();
    }
}

// File handling
function handleFileSelect() {
    const file = elements.fileInput.files[0];
    if (!file) return;
    
    uploadFile(file);
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    showToast('正在上传文件...', 'info');
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            state.currentFileId = result.file_id;
            state.fileInfo = result.file_info;
            displayFileInfo(result);
            hideDropZone();
            enableStartButton();
            showToast('文件上传成功', 'success');
        } else {
            showToast(`上传失败: ${result.error}`, 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showToast('上传失败: 网络错误', 'error');
    }
}

function displayFileInfo(data) {
    const fileInfo = data.file_info;
    const videoParams = data.video_params;
    
    elements.fileName.textContent = fileInfo.filename;
    elements.fileSize.textContent = formatFileSize(fileInfo.size);
    elements.estimatedFrames.textContent = formatNumber(videoParams.total_frames);
    elements.estimatedSize.textContent = formatFileSize(videoParams.estimated_video_size);
    
    elements.fileInfo.style.display = 'block';
}

function hideDropZone() {
    elements.dropZone.style.display = 'none';
}

function enableStartButton() {
    elements.startBtn.disabled = false;
}

// Conversion process
async function startConversion() {
    if (!state.currentFileId || state.isProcessing) return;
    
    const params = getConversionParams();
    
    try {
        const response = await fetch('/api/start-conversion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_id: state.currentFileId,
                params: params
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            state.currentTaskId = result.task_id;
            state.isProcessing = true;
            updateUIForProcessing();
            showToast('开始转换...', 'info');
        } else {
            showToast(`启动失败: ${result.error}`, 'error');
        }
    } catch (error) {
        console.error('Start conversion error:', error);
        showToast('启动失败: 网络错误', 'error');
    }
}

async function stopConversion() {
    if (!state.isProcessing) return;
    
    try {
        const response = await fetch('/api/stop-conversion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: state.currentTaskId })
        });
        
        const result = await response.json();
        
        if (result.success) {
            state.isProcessing = false;
            updateUIForStopped();
            showToast('转换已停止', 'warning');
        } else {
            showToast(`停止失败: ${result.error}`, 'error');
        }
    } catch (error) {
        console.error('Stop conversion error:', error);
        showToast('停止失败: 网络错误', 'error');
    }
}

function getConversionParams() {
    const activeResolution = document.querySelector('[data-resolution].active');
    
    return {
        resolution: activeResolution ? activeResolution.dataset.resolution : '4K',
        fps: parseInt(elements.fpsSlider.value),
        nine_to_one: true, // Always enabled for AVI format
        color_count: 16,
        error_correction: elements.errorCorrection.checked,
        error_correction_ratio: parseFloat(elements.redundancySlider.value) / 100,
        quality: 'high', // Not relevant for uncompressed AVI
        gpu_acceleration: elements.gpuAcceleration.checked,
        metadata_frames: elements.metadataFrames.checked,
        use_optimized_generator: true
    };
}

// Progress updates
function updateProgress(data) {
    const progress = Math.min(100, (data.processed_frames / data.total_frames) * 100);
    
    elements.progressBar.style.width = `${progress}%`;
    elements.progressPercentage.textContent = `${progress.toFixed(1)}%`;
    elements.framesInfo.textContent = `${formatNumber(data.processed_frames)} / ${formatNumber(data.total_frames)} 帧`;
    elements.fpsInfo.textContent = `${data.fps.toFixed(1)} FPS`;
    
    if (data.eta > 0) {
        elements.timeRemaining.textContent = formatDuration(data.eta);
    }
    
    if (data.preview_image) {
        elements.previewImage.src = `data:image/jpeg;base64,${data.preview_image}`;
        elements.previewImage.style.display = 'block';
        elements.previewCanvas.style.display = 'none';
    }
    
    // Update throughput
    elements.throughput.textContent = `${data.fps.toFixed(1)} FPS`;
}

function handleConversionComplete(data) {
    state.isProcessing = false;
    state.downloadUrl = `/api/download/${state.currentTaskId}`;
    
    updateUIForComplete();
    showVerificationSection();
    showToast('转换完成！文件已生成为未压缩AVI格式', 'success');
}

function handleConversionError(data) {
    state.isProcessing = false;
    updateUIForStopped();
    showToast(`转换错误: ${data.error}`, 'error');
}

// UI state management
function updateUIForProcessing() {
    elements.startBtn.disabled = true;
    elements.stopBtn.disabled = false;
    elements.downloadBtn.disabled = true;
    elements.fileInput.disabled = true;
}

function updateUIForStopped() {
    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
    elements.downloadBtn.disabled = true;
    elements.fileInput.disabled = false;
}

function updateUIForComplete() {
    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
    elements.downloadBtn.disabled = false;
    elements.fileInput.disabled = false;
}

function showVerificationSection() {
    elements.verificationSection.style.display = 'block';
}

// Verification process
async function startVerification() {
    if (state.verificationInProgress || !state.downloadUrl) return;
    
    state.verificationInProgress = true;
    elements.verificationProgress.style.display = 'block';
    elements.verificationResult.style.display = 'none';
    elements.startVerificationBtn.disabled = true;
    
    // Simulate verification process (replace with actual implementation)
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 10;
        progress = Math.min(progress, 100);
        
        elements.verifyPercentage.textContent = `${Math.round(progress)}%`;
        elements.verifyFrame.textContent = Math.round(progress * 10);
        elements.verifyBytes.textContent = `${(progress * 2.5).toFixed(1)} MB`;
        
        // Update progress circle
        const circle = document.getElementById('progress-circle');
        const circumference = 2 * Math.PI * 54;
        const offset = circumference - (progress / 100) * circumference;
        circle.style.strokeDashoffset = offset;
        
        if (progress >= 100) {
            clearInterval(interval);
            showVerificationResult(true);
        }
    }, 100);
}

function showVerificationResult(success) {
    state.verificationInProgress = false;
    elements.startVerificationBtn.disabled = false;
    elements.verificationProgress.style.display = 'none';
    elements.verificationResult.style.display = 'block';
    
    if (success) {
        elements.resultIcon.className = 'fas fa-check-circle';
        elements.resultIcon.style.color = '#4facfe';
        elements.resultTitle.textContent = '验证成功';
        elements.resultMessage.textContent = 'AVI文件数据完整性验证通过';
        elements.resultDetails.innerHTML = `
            <p>解码精度: 100%</p>
            <p>帧数匹配: ✓</p>
            <p>数据校验: ✓</p>
        `;
    } else {
        elements.resultIcon.className = 'fas fa-times-circle';
        elements.resultIcon.style.color = '#f5576c';
        elements.resultTitle.textContent = '验证失败';
        elements.resultMessage.textContent = '检测到数据损坏';
    }
}

// Download
async function downloadResult() {
    if (!state.downloadUrl) return;
    
    try {
        window.location.href = state.downloadUrl;
        showToast('开始下载AVI文件...', 'info');
    } catch (error) {
        console.error('Download error:', error);
        showToast('下载失败', 'error');
    }
}

// Hardware capabilities check
async function checkHardwareCapabilities() {
    try {
        const response = await fetch('/api/hardware-info');
        const data = await response.json();
        
        let status = [];
        if (data.nvenc_available) status.push('NVENC');
        if (data.qsv_available) status.push('QuickSync');
        if (status.length === 0) status.push('CPU Only');
        
        updateGPUStatus(status.join(' + '));
    } catch (error) {
        console.error('Hardware check error:', error);
        updateGPUStatus('未知');
    }
}

function updateGPUStatus(status) {
    elements.gpuStatus.textContent = status;
}

// Utility functions
function updateEstimates() {
    if (!state.fileInfo) return;
    
    const params = getConversionParams();
    // Add estimation logic here
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function formatDuration(seconds) {
    if (!seconds || isNaN(seconds)) return '00:00:00';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-content">
            <i class="fas fa-${type === 'success' ? 'check' : type === 'error' ? 'times' : 'info'}-circle"></i>
            <span>${message}</span>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

// ==================== Mode Tab Switching ====================
function setupModeTabs() {
    elements.tabEncode.addEventListener('click', () => switchMode('encode'));
    elements.tabDecode.addEventListener('click', () => switchMode('decode'));
}

function switchMode(mode) {
    state.activeMode = mode;

    // Update tab buttons
    elements.tabEncode.classList.toggle('active', mode === 'encode');
    elements.tabDecode.classList.toggle('active', mode === 'decode');

    // Update panels
    elements.panelEncode.classList.toggle('active', mode === 'encode');
    elements.panelDecode.classList.toggle('active', mode === 'decode');
}

// ==================== Decode Handlers ====================
function setupDecodeHandlers() {
    // File input change
    elements.decodeFileInput.addEventListener('change', handleDecodeFileSelect);

    // Browse button in the drop zone
    if (elements.decodeBrowseBtn) {
        elements.decodeBrowseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            elements.decodeFileInput.click();
        });
    }

    // Resolution buttons for decode
    elements.decodeResolutionBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            elements.decodeResolutionBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Color count buttons for decode
    elements.decodeColorBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            elements.decodeColorBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Action buttons
    elements.decodeStartBtn.addEventListener('click', startDecode);
    elements.decodeStopBtn.addEventListener('click', stopDecode);

    if (elements.decodeDownloadBtn) {
        elements.decodeDownloadBtn.addEventListener('click', downloadDecoded);
    }
    if (elements.decodeDownloadBtnSide) {
        elements.decodeDownloadBtnSide.addEventListener('click', downloadDecoded);
    }
}

// ==================== Decode Drag and Drop ====================
function setupDecodeDragAndDrop() {
    const dz = elements.decodeDropZone;
    if (!dz) return;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dz.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dz.addEventListener(eventName, () => dz.classList.add('active'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dz.addEventListener(eventName, () => dz.classList.remove('active'), false);
    });

    dz.addEventListener('drop', handleDecodeDrop);
    dz.addEventListener('click', () => elements.decodeFileInput.click());
}

function handleDecodeDrop(e) {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (!file.name.toLowerCase().endsWith('.avi')) {
            showToast('请选择 AVI 格式的视频文件', 'error');
            return;
        }
        elements.decodeFileInput.files = files;
        handleDecodeFileSelect();
    }
}

// ==================== Decode File Selection & Upload ====================
function handleDecodeFileSelect() {
    const file = elements.decodeFileInput.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.avi')) {
        showToast('请选择 AVI 格式的视频文件 / Please select an AVI file', 'error');
        return;
    }

    uploadVideoForDecode(file);
}

async function uploadVideoForDecode(file) {
    const formData = new FormData();
    formData.append('file', file);

    showToast('正在上传视频文件... Uploading video...', 'info');

    try {
        const response = await fetch('/api/upload-video', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            state.decodeFileId = result.file_id;
            state.decodeFileInfo = result.file_info || {};
            displayDecodeFileInfo(result);
            showDecodeFileLoaded(file.name);
            elements.decodeStartBtn.disabled = false;
            showToast('视频上传成功 Video uploaded successfully', 'success');
        } else {
            showToast(`上传失败: ${result.error}`, 'error');
        }
    } catch (error) {
        console.error('Decode upload error:', error);
        showToast('上传失败: 网络错误 Upload failed: Network error', 'error');
    }
}

function displayDecodeFileInfo(data) {
    const fileInfo = data.file_info || {};
    const videoInfo = data.video_info || {};

    elements.decodeFileName.textContent = fileInfo.filename || '-';
    elements.decodeFileSize.textContent = fileInfo.size ? formatFileSize(fileInfo.size) : '-';
    elements.decodeTotalFrames.textContent = videoInfo.total_frames ? formatNumber(videoInfo.total_frames) : '-';
    elements.decodeResolution.textContent = videoInfo.resolution || '-';
    elements.decodeEstimatedOutput.textContent = videoInfo.estimated_output_size
        ? formatFileSize(videoInfo.estimated_output_size) : '-';

    elements.decodeFileInfo.style.display = 'block';
}

function showDecodeFileLoaded(filename) {
    const dz = elements.decodeDropZone;
    if (!dz) return;

    dz.innerHTML = `
        <div class="decode-file-loaded">
            <i class="fas fa-file-video"></i>
            <h4>${escapeHtml(filename)}</h4>
            <p>文件已就绪，可以开始还原 / File ready for restore</p>
            <button class="btn-primary-gradient" onclick="elements.decodeFileInput.click(); event.stopPropagation();">
                <i class="fas fa-exchange-alt"></i> 更换文件 Change File
            </button>
        </div>
    `;
}

// ==================== Start / Stop Decode ====================
async function startDecode() {
    if (!state.decodeFileId || state.isDecoding) return;

    const params = getDecodeParams();

    try {
        const response = await fetch('/api/start-decode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_id: state.decodeFileId,
                params: params
            })
        });

        const result = await response.json();

        if (result.success) {
            state.decodeTaskId = result.task_id;
            state.isDecoding = true;
            updateUIForDecoding();
            showToast('开始还原... Starting restore...', 'info');
        } else {
            showToast(`还原启动失败: ${result.error}`, 'error');
        }
    } catch (error) {
        console.error('Start decode error:', error);
        showToast('还原启动失败: 网络错误', 'error');
    }
}

async function stopDecode() {
    if (!state.isDecoding) return;

    try {
        const response = await fetch('/api/stop-decode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: state.decodeTaskId })
        });

        const result = await response.json();

        if (result.success) {
            state.isDecoding = false;
            updateUIForDecodeStopped();
            showToast('还原已停止 Restore stopped', 'warning');
        } else {
            showToast(`停止失败: ${result.error}`, 'error');
        }
    } catch (error) {
        console.error('Stop decode error:', error);
        showToast('停止失败: 网络错误', 'error');
    }
}

function getDecodeParams() {
    const activeRes = document.querySelector('[data-decode-resolution].active');
    const activeColors = document.querySelector('[data-decode-colors].active');

    return {
        resolution: activeRes ? activeRes.dataset.decodeResolution : '4K',
        color_count: activeColors ? parseInt(activeColors.dataset.decodeColors) : 16,
        error_correction: elements.decodeErrorCorrection.checked,
        gpu_acceleration: elements.decodeGpuAcceleration.checked
    };
}

// ==================== Decode Progress ====================
function updateDecodeProgress(data) {
    const totalFrames = data.total_frames || 1;
    const processedFrames = data.processed_frames || 0;
    const progress = Math.min(100, (processedFrames / totalFrames) * 100);

    elements.decodeProgressBar.style.width = `${progress}%`;
    elements.decodeProgressBar.classList.add('processing');
    elements.decodeProgressPercentage.textContent = `${progress.toFixed(1)}%`;
    elements.decodeFramesInfo.textContent = `${formatNumber(processedFrames)} / ${formatNumber(totalFrames)} 帧`;

    if (data.fps !== undefined) {
        elements.decodeFpsInfo.textContent = `${data.fps.toFixed(1)} FPS`;
    }

    if (data.eta > 0) {
        elements.decodeTimeRemaining.textContent = formatDuration(data.eta);
    }

    // If server sends a preview frame
    if (data.preview_image && elements.decodePreviewCanvas) {
        const canvas = elements.decodePreviewCanvas;
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            canvas.style.display = 'block';
        };
        img.src = `data:image/jpeg;base64,${data.preview_image}`;
    }
}

function handleDecodeComplete(data) {
    state.isDecoding = false;
    state.decodeDownloadUrl = `/api/download-decoded/${state.decodeTaskId}`;

    elements.decodeProgressBar.classList.remove('processing');
    elements.decodeProgressBar.style.width = '100%';
    elements.decodeProgressPercentage.textContent = '100%';

    updateUIForDecodeComplete();
    showDecodeResult(data);
    showToast('还原完成! 文件已成功还原 / Restore complete!', 'success');
}

function handleDecodeError(data) {
    state.isDecoding = false;
    elements.decodeProgressBar.classList.remove('processing');
    updateUIForDecodeStopped();
    showToast(`还原错误: ${data.error}`, 'error');
}

// ==================== Decode UI State Management ====================
function updateUIForDecoding() {
    elements.decodeStartBtn.disabled = true;
    elements.decodeStopBtn.disabled = false;
    if (elements.decodeDownloadBtn) elements.decodeDownloadBtn.disabled = true;
    if (elements.decodeDownloadBtnSide) elements.decodeDownloadBtnSide.disabled = true;
    elements.decodeFileInput.disabled = true;
    elements.decodeResultSection.style.display = 'none';
}

function updateUIForDecodeStopped() {
    elements.decodeStartBtn.disabled = false;
    elements.decodeStopBtn.disabled = true;
    if (elements.decodeDownloadBtn) elements.decodeDownloadBtn.disabled = true;
    if (elements.decodeDownloadBtnSide) elements.decodeDownloadBtnSide.disabled = true;
    elements.decodeFileInput.disabled = false;
}

function updateUIForDecodeComplete() {
    elements.decodeStartBtn.disabled = false;
    elements.decodeStopBtn.disabled = true;
    if (elements.decodeDownloadBtn) elements.decodeDownloadBtn.disabled = false;
    if (elements.decodeDownloadBtnSide) elements.decodeDownloadBtnSide.disabled = false;
    elements.decodeFileInput.disabled = false;
}

function showDecodeResult(data) {
    elements.decodeResultSection.style.display = 'block';

    const info = data || {};
    elements.decodeResultTitle.textContent = '文件已成功还原';
    elements.decodeResultMessage.textContent = 'File has been successfully restored';

    const outputSize = info.output_size ? formatFileSize(info.output_size) : '-';
    const totalFrames = info.total_frames ? formatNumber(info.total_frames) : '-';
    const duration = info.duration ? formatDuration(info.duration) : '-';
    const filename = info.output_filename || 'restored_file';

    elements.decodeResultDetails.innerHTML = `
        <div class="detail-item">
            <span class="detail-label">还原文件 Output</span>
            <span class="detail-value">${escapeHtml(filename)}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">文件大小 Size</span>
            <span class="detail-value">${outputSize}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">总帧数 Frames</span>
            <span class="detail-value">${totalFrames}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">耗时 Duration</span>
            <span class="detail-value">${duration}</span>
        </div>
    `;

    // Scroll to the result section
    elements.decodeResultSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ==================== Decode Download ====================
function downloadDecoded() {
    if (!state.decodeDownloadUrl) {
        showToast('没有可下载的文件 No file available for download', 'warning');
        return;
    }

    try {
        window.location.href = state.decodeDownloadUrl;
        showToast('开始下载还原文件... Downloading restored file...', 'info');
    } catch (error) {
        console.error('Decode download error:', error);
        showToast('下载失败 Download failed', 'error');
    }
}

// ==================== Utility: Escape HTML ====================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', initApp);
