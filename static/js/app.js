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

// Progress tracking
class ProgressTracker {
    constructor() {
        this.currentProgress = 0;
        this.targetProgress = 0;
        this.animationId = null;
    }

    updateProgress(target, message = '') {
        this.targetProgress = target;
        
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
            if (message) progressText.textContent = message;
            return;
        }
        
        this.currentProgress += (this.targetProgress - this.currentProgress) * 0.1;
        progressBar.style.width = this.currentProgress + '%';
        if (message) progressText.textContent = message;
        
        this.animationId = requestAnimationFrame(() => this.animateProgress(message));
    }

    reset() {
        this.currentProgress = 0;
        this.targetProgress = 0;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
}

// Log manager
class LogManager {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.maxEntries = 100;
    }

    addEntry(message, type = 'info') {
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        
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
            progressTracker.updateProgress(100, 'Processing completed!');
            logManager.addEntry('Processing completed successfully!', 'success');
            
            // Display results
            displayResults(data);
        } else {
            throw new Error(data.error || 'Processing failed');
        }
    } catch (error) {
        console.error('Error processing document:', error);
        logManager.addEntry(`Error: ${error.message}`, 'error');
        progressTracker.updateProgress(0, 'Processing failed');
        Utils.showNotification(`Processing failed: ${error.message}`, 'danger');
    } finally {
        processingInProgress = false;
        processBtn.disabled = false;
        processBtn.classList.remove('btn-loading');
        processBtn.innerHTML = '<i class="fas fa-play me-2"></i>Process Document';
    }
}

function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    const downloadLinks = document.getElementById('downloadLinks');
    
    downloadLinks.innerHTML = '';
    
    if (data.word_file) {
        const downloadBtn = document.createElement('a');
        downloadBtn.href = `/api/download/${data.word_file}`;
        downloadBtn.className = 'btn btn-success me-2';
        downloadBtn.innerHTML = '<i class="fas fa-download me-2"></i>Download Word Document';
        downloadLinks.appendChild(downloadBtn);
        
        // Add debug logging
        console.log('Download link created:', `/api/download/${data.word_file}`);
    }
    
    // Add log entries
    if (data.log) {
        data.log.forEach(log => logManager.addEntry(log));
    }
    
    resultsSection.style.display = 'block';
    Utils.showNotification('Document processed successfully!', 'success');
}

// Global variable for processing state
let processingInProgress = false;

// Export functions for global access
window.processDocument = processDocument; 