# PDF/Image to Word Converter - Flask Web Application

A modern web-based interface for converting PDF and image documents to Word format with advanced OCR and layout analysis capabilities.

## Features

- **Web-based Interface**: Modern, responsive web UI built with Flask and Bootstrap
- **File Upload**: Drag-and-drop or click-to-browse file upload
- **Multiple Formats**: Support for PDF, JPG, JPEG, PNG, BMP, and TIFF files
- **OCR Processing**: Advanced text recognition with configurable confidence thresholds
- **Layout Analysis**: YOLO-based layout detection for tables, figures, headers, etc.
- **Real-time Progress**: Live progress tracking and detailed processing logs
- **Download Results**: Direct download of processed Word documents
- **Settings Configuration**: Adjustable DPI, language, and processing options

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd intellidoc
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model weights are available**:
   - Make sure the `weights` and `rec2` folder exists

## Usage

### Starting the Flask Application

1. **Run the Flask app**:
   ```bash
   python flask_app.py
   ```

2. **Access the web interface**:
   - Open your browser and navigate to `http://localhost:5000`
   - The application will automatically initialize the required models

### Using the Web Interface

1. **System Status**: The application shows the current status of model loading
2. **File Upload**: 
   - Drag and drop your file into the upload area, or
   - Click "Browse Files" to select a file
3. **Configure Settings**:
   - **Language**: Select the document language (Sinhala, English, etc.)
   - **PDF DPI**: Set resolution for PDF conversion (150-600)
   - **OCR Confidence**: Adjust confidence threshold (50-100%)
   - **Save Options**: Choose to save intermediate JSON files and processed images
4. **Process Document**: Click "Process Document" to start conversion
5. **Monitor Progress**: Watch real-time progress and detailed logs
6. **Download Results**: Download the converted Word document when processing completes

## File Structure

```
intellidoc/
├── flask_app.py              # Main Flask application
├── templates/
│   └── index.html            # Main web interface template
├── static/
│   ├── css/
│   │   └── style.css         # Additional CSS styles
│   └── js/
│       └── app.js            # JavaScript functionality
├── uploads/                  # Temporary file upload directory
├── outputs/                  # Processed file output directory
├── weights/
│   └── best.pt              # YOLO layout detection model
└── requirements.txt          # Python dependencies
```

## API Endpoints

- `GET /` - Main web interface
- `GET /api/status` - Check system status
- `POST /api/initialize` - Initialize models
- `POST /api/upload` - Upload and process files
- `GET /api/download/<filename>` - Download processed files

## Configuration

### Environment Variables

The application uses the same environment variables as the original Tkinter app:

```bash
export RECOGNITION_BATCH_SIZE=256
export DETECTOR_BATCH_SIZE=18
export ORDER_BATCH_SIZE=16
export RECOGNITION_STATIC_CACHE=true
```

### Flask Configuration

Key Flask settings in `flask_app.py`:

- **Max file size**: 100MB
- **Upload folder**: `uploads/`
- **Output folder**: `outputs/`
- **Debug mode**: Enabled for development

## Browser Compatibility

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Troubleshooting

### Common Issues

1. **Models not loading**:
   - Check internet connection for model downloads
   - Ensure sufficient disk space
   - Verify GPU/CPU compatibility

2. **File upload fails**:
   - Check file size (max 100MB)
   - Verify file format is supported
   - Ensure upload directory has write permissions

3. **Processing errors**:
   - Check the processing log for detailed error messages
   - Verify model weights are present
   - Ensure sufficient system memory

### Performance Optimization

- **For large files**: Increase system memory allocation
- **For faster processing**: Use GPU acceleration if available
- **For multiple users**: Consider using a production WSGI server

## Development

### Adding New Features

1. **Backend**: Modify `flask_app.py` for new API endpoints
2. **Frontend**: Update `templates/index.html` for UI changes
3. **Styling**: Modify `static/css/style.css` for visual updates
4. **Functionality**: Update `static/js/app.js` for client-side features

### Testing

```bash
# Run with debug mode
python flask_app.py

# Test API endpoints
curl http://localhost:5000/api/status
```

## Production Deployment

For production deployment, consider:

1. **WSGI Server**: Use Gunicorn or uWSGI
2. **Reverse Proxy**: Configure Nginx or Apache
3. **Environment**: Set `FLASK_ENV=production`
4. **Security**: Change default secret key
5. **Monitoring**: Add logging and monitoring

Example Gunicorn deployment:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 flask_app:app
```

## License

This project maintains the same license as the original application.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the processing logs
3. Ensure all dependencies are properly installed
4. Verify model files are present and accessible 