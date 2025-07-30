// Additional JavaScript functionality for the Flask web application

// Utility functions
const Utils = {
    // Format file size
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    // Debounce function
    debounce: function(func, wait, immediate) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                timeout = null;
                if (!immediate) func(...args);
            };
            const callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func(...args);
        };
    },

    // Show notification
    showNotification: function(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    },

    // Validate file type
    validateFile: function(file) {
        const allowedTypes = ['application/pdf', 'image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff'];
        const maxSize = 100 * 1024 * 1024; // 100MB
        
        if (!allowedTypes.includes(file.type)) {
            return { valid: false, error: 'File type not supported. Please upload PDF or image files.' };
        }
        
        if (file.size > maxSize) {
            return { valid: false, error: 'File size too large. Maximum size is 100MB.' };
        }
        
        return { valid: true };
    }
};

// Progress tracking with detailed step tracking
class ProgressTracker {
    constructor() {
        this.currentProgress = 0;
        this.targetProgress = 0;
        this.animationId = null;
        this.currentStep = '';
        this.currentPage = null;
        this.totalPages = null;
        this.eventSource = null;
    }

    updateProgress(target, message = '', step = '') {
        this.targetProgress = target;
        this.currentStep = step;
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        this.animateProgress(message);
    }

    animateProgress(message) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        
        if (Math.abs(this.currentProgress - this.targetProgress) < 1) {
            this.currentProgress = this.targetProgress;
            progressBar.style.width = this.currentProgress + '%';
            this.updateProgressText(message);
            return;
        }
        
        this.currentProgress += (this.targetProgress - this.currentProgress) * 0.1;
        progressBar.style.width = this.currentProgress + '%';
        this.updateProgressText(message);
        
        this.animationId = requestAnimationFrame(() => this.animateProgress(message));
    }

    updateProgressText(message) {
        const progressText = document.getElementById('progressText');
        let displayText = message;
        
        if (this.currentPage && this.totalPages) {
            displayText = `<span class="page-progress">Page ${this.currentPage}/${this.totalPages}</span> ${message}`;
        }
        
        if (this.currentStep) {
            displayText = `<span class="step-indicator">${this.currentStep}</span> ${displayText}`;
        }
        
        progressText.innerHTML = displayText;
    }

    startProgressStream(taskId) {
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        // Show processing status indicator
        document.getElementById('processingStatus').style.display = 'flex';
        
        this.eventSource = new EventSource(`/api/progress/${taskId}`);
        
        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleProgressUpdate(data);
            } catch (error) {
                console.error('Error parsing progress update:', error);
            }
        };
        
        this.eventSource.onerror = (error) => {
            console.error('Progress stream error:', error);
            this.eventSource.close();
        };
    }

    handleProgressUpdate(data) {
        const { type, message, progress, page_num, total_pages } = data;
        
        // Update page info
        if (page_num !== undefined) this.currentPage = page_num;
        if (total_pages !== undefined) this.totalPages = total_pages;
        
        // Map progress types to step names
        const stepNameMap = {
            'file_analysis': 'File Analysis',
            'pdf_conversion': 'PDF Conversion',
            'processing_start': 'Processing',
            'page_start': 'Page Processing',
            'ocr_start': 'OCR',
            'ocr_progress': 'OCR Processing',
            'ocr_complete': 'OCR Complete',
            'layout_start': 'Layout Analysis',
            'layout_progress': 'Layout Processing',
            'layout_complete': 'Layout Complete',
            'saving_intermediate': 'Saving Files',
            'saving_images': 'Saving Images',
            'word_generation': 'Word Generation',
            'complete': 'Complete',
            'error': 'Error',
            'warning': 'Warning'
        };
        
        const stepName = stepNameMap[type] || type;
        
        // Update progress
        if (progress !== undefined) {
            this.updateProgress(progress, message, stepName);
        } else {
            this.updateProgressText(message);
        }
        
        // Add to log with step information
        const logStepMap = {
            'ocr_start': 'ocr',
            'ocr_progress': 'ocr',
            'ocr_complete': 'ocr',
            'ocr_error': 'error',
            'layout_start': 'layout',
            'layout_progress': 'layout',
            'layout_complete': 'layout',
            'layout_error': 'error',
            'word_generation': 'word',
            'complete': 'success',
            'error': 'error'
        };
        
        const step = logStepMap[type] || '';
        logManager.addEntry(message, this.getLogType(type), step);
        
        // Handle completion
        if (type === 'complete') {
            this.handleCompletion(data);
        } else if (type === 'error') {
            this.handleError(data);
        }
    }

    getLogType(type) {
        if (type === 'error') return 'error';
        if (type === 'warning') return 'warning';
        if (type === 'complete') return 'success';
        return 'info';
    }

    handleCompletion(data) {
        console.log('Completion data received:', data);
        
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        // Hide processing status indicator
        document.getElementById('processingStatus').style.display = 'none';
        
        // Show completion notification
        Utils.showNotification('Processing completed successfully!', 'success');
        
        // Enable process button
        const processBtn = document.getElementById('processBtn');
        processBtn.disabled = false;
        processBtn.innerHTML = '<i class="fas fa-play me-2"></i>Process Document';
        
        // Show results section
        document.getElementById('resultsSection').style.display = 'block';
        
        // Create download link if word file is available
        if (data.word_file) {
            console.log('Creating download link for:', data.word_file);
            this.createDownloadLink(data.word_file);
        } else {
            console.log('No word_file in completion data');
        }
    }

    handleError(data) {
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        // Hide processing status indicator
        document.getElementById('processingStatus').style.display = 'none';
        
        // Show error notification
        Utils.showNotification(`Processing failed: ${data.message}`, 'danger');
        
        // Enable process button
        const processBtn = document.getElementById('processBtn');
        processBtn.disabled = false;
        processBtn.innerHTML = '<i class="fas fa-play me-2"></i>Process Document';
    }

    createDownloadLink(filename) {
        console.log('Creating download link with filename:', filename);
        const downloadLinks = document.getElementById('downloadLinks');
        downloadLinks.innerHTML = '';
        
        const downloadBtn = document.createElement('a');
        downloadBtn.href = `/api/download/${filename}`;
        downloadBtn.className = 'btn btn-success me-2';
        downloadBtn.innerHTML = '<i class="fas fa-download me-2"></i>Download Word Document';
        downloadLinks.appendChild(downloadBtn);
        
        console.log('Download button created:', downloadBtn);
        console.log('Download links container:', downloadLinks);
    }

    reset() {
        this.currentProgress = 0;
        this.targetProgress = 0;
        this.currentStep = '';
        this.currentPage = null;
        this.totalPages = null;
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        if (this.eventSource) {
            this.eventSource.close();
        }
    }
}

// Log manager
class LogManager {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.maxEntries = 100;
    }

    addEntry(message, type = 'info', step = '') {
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        
        // Add step-specific styling
        if (step) {
            entry.setAttribute('data-step', step);
        }
        
        if (type === 'error') {
            entry.style.borderLeftColor = '#dc3545';
            entry.style.backgroundColor = '#f8d7da';
        } else if (type === 'success') {
            entry.style.borderLeftColor = '#28a745';
            entry.style.backgroundColor = '#d4edda';
        } else if (type === 'warning') {
            entry.style.borderLeftColor = '#ffc107';
            entry.style.backgroundColor = '#fff3cd';
        }
        
        const timestamp = new Date().toLocaleTimeString();
        entry.innerHTML = `<strong>[${timestamp}]</strong> ${message}`;
        
        this.container.appendChild(entry);
        
        // Limit number of entries
        while (this.container.children.length > this.maxEntries) {
            this.container.removeChild(this.container.firstChild);
        }
        
        this.scrollToBottom();
    }

    clear() {
        this.container.innerHTML = '';
    }

    scrollToBottom() {
        this.container.scrollTop = this.container.scrollHeight;
    }
}

// File upload manager
class FileUploadManager {
    constructor() {
        this.selectedFile = null;
        this.setupEventListeners();
    }

    setupEventListeners() {
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('fileUploadArea');
        const browseBtn = document.getElementById('browseBtn');
        
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Separate event listeners to avoid conflicts
        browseBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            fileInput.click();
        });
        
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        
        // Only trigger file input on direct click to upload area (not on button)
        uploadArea.addEventListener('click', (e) => {
            if (e.target === uploadArea || e.target.classList.contains('file-upload-area')) {
                fileInput.click();
            }
        });
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        event.currentTarget.classList.add('dragover');
    }

    handleDragLeave(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('dragover');
    }

    handleDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('dragover');
        
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    processFile(file) {
        const validation = Utils.validateFile(file);
        
        if (!validation.valid) {
            Utils.showNotification(validation.error, 'danger');
            return;
        }
        
        this.selectedFile = file;
        this.displayFileInfo(file);
        logManager.addEntry(`File selected: ${file.name} (${Utils.formatFileSize(file.size)})`, 'success');
    }

    displayFileInfo(file) {
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        
        fileName.textContent = `${file.name} (${Utils.formatFileSize(file.size)})`;
        fileInfo.style.display = 'block';
    }

    getSelectedFile() {
        return this.selectedFile;
    }

    clearSelection() {
        this.selectedFile = null;
        document.getElementById('fileInfo').style.display = 'none';
        document.getElementById('fileInput').value = '';
    }
}

// API manager
class ApiManager {
    static async checkStatus() {
        try {
            const response = await fetch('/api/status');
            return await response.json();
        } catch (error) {
            console.error('Error checking status:', error);
            throw error;
        }
    }

    static async initializeModels() {
        try {
            const response = await fetch('/api/initialize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            return await response.json();
        } catch (error) {
            console.error('Error initializing models:', error);
            throw error;
        }
    }

    static async uploadFile(formData) {
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            return await response.json();
        } catch (error) {
            console.error('Error uploading file:', error);
            throw error;
        }
    }
}

// Global instances
let progressTracker;
let logManager;
let fileUploadManager;

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    progressTracker = new ProgressTracker();
    logManager = new LogManager('logContainer');
    fileUploadManager = new FileUploadManager();
    
    // Add process button event listener
    document.getElementById('processBtn').addEventListener('click', processDocument);
    
    // Start status checking
    checkSystemStatus();
});

// System status management
async function checkSystemStatus() {
    try {
        const status = await ApiManager.checkStatus();
        updateStatus(status.status, status.message);
        
        if (status.status === 'not_initialized') {
            await initializeModels();
        }
    } catch (error) {
        updateStatus('error', 'Failed to check system status');
        logManager.addEntry('Failed to check system status', 'error');
    }
}

async function initializeModels() {
    try {
        const result = await ApiManager.initializeModels();
        updateStatus(result.status, result.message);
        
        if (result.status === 'loading') {
            // Poll for status updates
            setTimeout(checkSystemStatus, 2000);
        }
    } catch (error) {
        updateStatus('error', 'Failed to initialize models');
        logManager.addEntry('Failed to initialize models', 'error');
    }
}

function updateStatus(status, message) {
    const indicator = document.getElementById('statusIndicator');
    const statusMessage = document.getElementById('statusMessage');
    const processBtn = document.getElementById('processBtn');
    const alert = document.getElementById('statusAlert');
    
    // Update indicator
    indicator.className = 'status-indicator';
    if (status === 'loading') {
        indicator.classList.add('status-loading');
    } else if (status === 'ready') {
        indicator.classList.add('status-ready');
        processBtn.disabled = false;
    } else if (status === 'error') {
        indicator.classList.add('status-error');
    }
    
    // Update message
    statusMessage.textContent = message;
    
    // Update alert class
    alert.className = 'alert';
    if (status === 'ready') {
        alert.classList.add('alert-success');
    } else if (status === 'loading') {
        alert.classList.add('alert-warning');
    } else if (status === 'error') {
        alert.classList.add('alert-danger');
    } else {
        alert.classList.add('alert-info');
    }
}

// Document processing
async function processDocument() {
    const selectedFile = fileUploadManager.getSelectedFile();
    
    if (!selectedFile) {
        Utils.showNotification('Please select a file first', 'warning');
        return;
    }
    
    if (processingInProgress) return;
    
    processingInProgress = true;
    const processBtn = document.getElementById('processBtn');
    processBtn.disabled = true;
    processBtn.classList.add('btn-loading');
    processBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
    
    // Show progress section
    document.getElementById('progressSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    
    // Reset progress tracker
    progressTracker.reset();
    
    // Clear log
    logManager.clear();
    
    // Create form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('language', document.getElementById('languageSelect').value);
    formData.append('dpi', document.getElementById('dpiInput').value);
    formData.append('confidence', document.getElementById('confidenceInput').value);
    formData.append('save_json', document.getElementById('saveJsonCheck').checked);
    formData.append('save_images', document.getElementById('saveImagesCheck').checked);
    
    logManager.addEntry('Starting document processing...', 'info');
    progressTracker.updateProgress(0, 'Initializing...');
    
    try {
        const data = await ApiManager.uploadFile(formData);
        
        if (data.success) {
            logManager.addEntry('Processing started, monitoring progress...', 'info');
            
            // Start progress stream
            progressTracker.startProgressStream(data.task_id);
        } else {
            throw new Error(data.error || 'Processing failed');
        }
    } catch (error) {
        console.error('Error processing document:', error);
        logManager.addEntry(`Error: ${error.message}`, 'error');
        progressTracker.updateProgress(0, 'Processing failed');
        Utils.showNotification(`Processing failed: ${error.message}`, 'danger');
        
        // Reset UI
        processingInProgress = false;
        processBtn.disabled = false;
        processBtn.classList.remove('btn-loading');
        processBtn.innerHTML = '<i class="fas fa-play me-2"></i>Process Document';
    }
}

// Global variable for processing state
let processingInProgress = false;

// Export functions for global access
window.processDocument = processDocument; 