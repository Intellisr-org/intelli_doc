import os
import json
import threading
import gc
import logging
from flask import Flask, render_template, request, jsonify, send_file, session, Response, stream_with_context
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import torch
import fitz  # PyMuPDF - simpler alternative to pdf2image
from ultralytics import YOLO
from datetime import datetime
import queue
import time
import psutil
# Import Surya models
from intellidoc.ocr import run_ocr
from intellidoc.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from intellidoc.model.recognition.model import load_model as load_rec_model
from intellidoc.model.recognition.processor import load_processor as load_rec_processor
from intellidoc.settings import settings
# Import Word document processor
from word_processor import WordDocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration
from config import config

# Flask app configuration
app = Flask(__name__)
app.config.from_object(config['default'])
config['default'].init_app(app)

# Allowed file extensions
ALLOWED_EXTENSIONS = app.config['ALLOWED_EXTENSIONS']

# Global progress tracking
progress_queues = {}

# Progress queue configuration
PROGRESS_QUEUE_MAXSIZE = 200  # Increased from 100
PROGRESS_QUEUE_TIMEOUT = 2    # Increased timeout

def cleanup_old_progress_queues():
    """Clean up old progress queues to prevent memory leaks"""
    current_time = time.time()
    expired_tasks = []
    
    for task_id in list(progress_queues.keys()):
        # Check if task is older than 1 hour (3600 seconds)
        try:
            task_timestamp = int(task_id.split('_')[1]) / 1000  # Convert from milliseconds
            if current_time - task_timestamp > 3600:  # 1 hour
                expired_tasks.append(task_id)
        except (IndexError, ValueError):
            # If task_id format is invalid, mark for cleanup
            expired_tasks.append(task_id)
    
    # Remove expired tasks
    for task_id in expired_tasks:
        try:
            del progress_queues[task_id]
            logger.info(f"Cleaned up expired progress queue: {task_id}")
        except KeyError:
            pass

def cleanup_memory():
    """Clean up GPU and CPU memory"""
    try:
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Clear Python memory
        import sys
        if hasattr(sys, 'exc_clear'):
            sys.exc_clear()
            
        logger.info("Memory cleanup completed")
    except Exception as e:
        logger.warning(f"Memory cleanup warning: {e}")

def get_memory_usage():
    """Get current memory usage information"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        memory_data = {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        }
        
        if torch.cuda.is_available():
            memory_data['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_data['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return memory_data
    except Exception as e:
        logger.warning(f"Could not get memory usage: {e}")
        return {}

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Image.Image):
            return "Image object (not serializable)"
        if hasattr(obj, '__dict__'):
            return {k: self.default(v) for k, v in obj.__dict__.items()}
        return str(obj)

def serialize_result(result):
    return json.dumps(result, cls=CustomJSONEncoder, indent=2)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def emit_progress(task_id, progress_type, message, progress_percentage=None, page_num=None, total_pages=None, word_file=None, memory_info=None):
    """Emit progress update to the client"""
    if task_id in progress_queues:
        progress_data = {
            'type': progress_type,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        if progress_percentage is not None:
            progress_data['progress'] = progress_percentage
            
        if page_num is not None:
            progress_data['page_num'] = page_num
            
        if total_pages is not None:
            progress_data['total_pages'] = total_pages
            
        if word_file is not None:
            progress_data['word_file'] = word_file
            logger.info(f"Adding word_file to progress data: {word_file}")
            
        if memory_info is not None:
            progress_data['memory_info'] = memory_info
            
        try:
            progress_queues[task_id].put(progress_data, timeout=PROGRESS_QUEUE_TIMEOUT)
            if progress_type == 'complete':
                logger.info(f"Sent completion progress data: {progress_data}")
        except queue.Full:
            logger.warning(f"Progress queue full for task {task_id}")
            # Try to clear some old messages to make room
            try:
                # Remove oldest message to make room for new one
                progress_queues[task_id].get_nowait()
                progress_queues[task_id].put(progress_data, timeout=1)
            except:
                pass  # If clearing fails, just log the warning

def draw_boxes(image, predictions, color=(255, 0, 0)):
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(image)
    if isinstance(predictions, list):
        for pred in predictions:
            if hasattr(pred, 'bboxes'):
                for bbox in pred.bboxes:
                    if hasattr(bbox, 'bbox'):
                        draw.rectangle(bbox.bbox, outline=color, width=2)
                    elif hasattr(bbox, 'polygon'):
                        draw.polygon(bbox.polygon, outline=color, width=2)
    elif hasattr(predictions, 'bboxes'):
        for bbox in predictions.bboxes:
            if hasattr(bbox, 'bbox'):
                draw.rectangle(bbox.bbox, outline=color, width=2)
            elif hasattr(bbox, 'polygon'):
                draw.polygon(bbox.polygon, outline=color, width=2)
    return image

class PDFWordConverter:
    def __init__(self, yolo_model_path):
        self.device = app.config['DEVICE']
        self.models_loaded = False
        self.det_model = None
        self.det_processor = None
        self.rec_model = None
        self.rec_processor = None
        self.layout_model = None
        self.layout_processor = None
        self.order_model = None
        self.yolo_model_path = yolo_model_path
        
    def cleanup_after_page(self):
        """Clean up memory after processing each page"""
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Page processing cleanup completed")
        except Exception as e:
            logger.warning(f"Page cleanup warning: {e}")
        
    def load_models(self):
        """Load all required models"""
        try:
            logger.info("Loading detection models...")
            self.det_processor, self.det_model = load_det_processor(), load_det_model()
            
            logger.info("Loading recognition models...")
            # Set static cache to True to avoid TorchDynamo compilation issues
            settings.RECOGNITION_STATIC_CACHE = True
            self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()
            
            logger.info("Loading YOLO layout models...")
            self.yolo_model = YOLO(self.yolo_model_path)
            self.yolo_model.to(self.device)
            
            # Disable model compilation for recognition model to avoid TorchDynamo issues
            logger.info("Recognition model loaded without compilation to avoid TorchDynamo issues")
            
            self.models_loaded = True
            logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def ocr_workflow(self, image, langs=['en'], page_num=1, task_id=None):
        """Perform OCR on a single page and return results with bounding boxes"""
        if not self.models_loaded:
            raise Exception("Models not loaded")
            
        try:
            if task_id:
                emit_progress(task_id, 'ocr_start', f"Starting OCR for page {page_num}", 0, page_num)
            
            logger.info(f"Starting OCR workflow for page {page_num}")
            
            # Clear memory before OCR
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            predictions = run_ocr([image], [langs], self.det_model, self.det_processor, 
                                self.rec_model, self.rec_processor)
            
            if task_id:
                emit_progress(task_id, 'ocr_progress', f"OCR processing completed for page {page_num}", 50, page_num)
            
            # Convert to serializable format
            page_result = {
                'page_number': page_num,
                'page_width': image.size[0],
                'page_height': image.size[1],
                'text_lines': [],
            }
            
            for line in predictions[0].text_lines:
                line_data = {
                    'text': line.text,
                    'bbox': line.bbox if hasattr(line, 'bbox') else None,
                    'polygon': line.polygon if hasattr(line, 'polygon') else None,
                    'confidence': line.confidence if hasattr(line, 'confidence') else None
                }
                page_result['text_lines'].append(line_data)
            
            # Clear predictions from memory
            del predictions
            
            if task_id:
                emit_progress(task_id, 'ocr_complete', f"OCR completed for page {page_num} - Found {len(page_result['text_lines'])} text lines", 100, page_num)
            
            logger.info(f"OCR completed for page {page_num}")
            return page_result
            
        except Exception as e:
            if task_id:
                emit_progress(task_id, 'ocr_error', f"OCR error on page {page_num}: {str(e)}", 0, page_num)
            logger.error(f"Error during OCR workflow: {e}")
            return {
                'page_number': page_num,
                'page_width': image.size[0] if image else 0,
                'page_height': image.size[1] if image else 0,
                'text_lines': [],
                'error': str(e)
            }

    def yolo_layout_analysis_workflow(self, image, page_num=1, task_id=None):
        """Perform layout analysis on a single page"""
        if not self.models_loaded:
            raise Exception("Models not loaded")
            
        try:
            if task_id:
                emit_progress(task_id, 'layout_start', f"Starting layout analysis for page {page_num}", 0, page_num)
            
            logger.info(f"Starting layout analysis for page {page_num}")

            id2label = {
                0: "Caption",
                1: "Footnote",
                2: "Formula",
                3: "List",
                4: "PageFooter",
                5: "PageHeader",
                6: "Figure",
                7: "SectionHeader",
                8: "Table",
                9: "TableofContents",
                10: "Text",
                11: "Title"
            }
            
            if task_id:
                emit_progress(task_id, 'layout_progress', f"Running YOLO detection for page {page_num}", 30, page_num)
            
            # Clear memory before YOLO prediction
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # YOLO prediction with optimized parameters to avoid NMS timeout
            results = self.yolo_model.predict(
                image, 
                save=False, 
                conf=0.7,  # Higher confidence threshold
                iou=0.5,   # IOU threshold for NMS
                max_det=50,  # Limit maximum detections
                verbose=False  # Reduce logging
            )
            
            if task_id:
                emit_progress(task_id, 'layout_progress', f"Processing layout results for page {page_num}", 70, page_num)
            
            # Convert to serializable format
            serializable_predictions = []
            for result in results:
                serializable_pred = {
                    'bboxes': [
                        {
                            'bbox': [int(x) for x in bbox.xyxy.tolist()[0]] if hasattr(bbox, 'xyxy') and bbox.xyxy is not None else None,
                            'polygon': self._convert_xyxy_to_polygon(bbox.xyxy.tolist()[0]) if hasattr(bbox, 'xyxy') and bbox.xyxy is not None else None,
                            'confidence': float(bbox.conf) if hasattr(bbox, 'conf') else 0.0,
                            'label': id2label[int(bbox.cls)] if hasattr(bbox, 'cls') else 0
                        }
                        for bbox in result.boxes
                    ] if hasattr(result, 'boxes') and result.boxes is not None else [],
                    'image_bbox': [0, 0, result.orig_shape[1], result.orig_shape[0]] if hasattr(result, 'orig_shape') else None
                }
                serializable_predictions.append(serializable_pred)
            
            # Clear results from memory
            del results
            
            if task_id:
                emit_progress(task_id, 'layout_complete', f"Layout analysis completed for page {page_num} - Found {len(serializable_predictions[0]['bboxes'])} layout elements", 100, page_num)
            
            logger.info(f"Layout analysis completed for page {page_num}")
            return serializable_predictions
            
        except Exception as e:
            if task_id:
                emit_progress(task_id, 'layout_error', f"Layout analysis error on page {page_num}: {str(e)}", 0, page_num)
            logger.error(f"Error during layout analysis: {e}")
            return [{'bboxes': [], 'image_bbox': None, 'error': str(e)}]
    
    def _convert_xyxy_to_polygon(self, xyxy):
        """Convert xyxy format [x1, y1, x2, y2] to polygon format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]"""
        if xyxy is None or len(xyxy) != 4:
            return None
        x1, y1, x2, y2 = xyxy
        return [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]

# Global converter instance
converter = None
models_loading = False

def initialize_converter():
    """Initialize the converter with models"""
    global converter, models_loading
    if converter is None and not models_loading:
        models_loading = True
        converter = PDFWordConverter(yolo_model_path=app.config['YOLO_MODEL_PATH'])
        success = converter.load_models()
        models_loading = False
        return success
    return converter is not None and converter.models_loaded

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get processing status"""
    global converter, models_loading
    
    if models_loading:
        return jsonify({'status': 'loading', 'message': 'Models are being loaded...'})
    elif converter is None:
        return jsonify({'status': 'not_initialized', 'message': 'Converter not initialized'})
    elif converter.models_loaded:
        return jsonify({'status': 'ready', 'message': 'Models loaded successfully! Ready to process documents.'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to load models'})

@app.route('/api/initialize', methods=['POST'])
def initialize_models():
    """Initialize models"""
    global converter, models_loading
    
    if models_loading:
        return jsonify({'status': 'loading', 'message': 'Models are already being loaded...'})
    
    if converter is None:
        models_loading = True
        
        def load_models_async():
            global converter, models_loading
            try:
                converter = PDFWordConverter(yolo_model_path="weights/best.pt")
                success = converter.load_models()
                models_loading = False
                logger.info(f"Models loaded: {success}")
            except Exception as e:
                models_loading = False
                logger.error(f"Error loading models: {e}")
        
        threading.Thread(target=load_models_async, daemon=True).start()
        return jsonify({'status': 'loading', 'message': 'Models are being loaded...'})
    
    return jsonify({'status': 'ready', 'message': 'Models already loaded'})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    global converter
    
    if converter is None or not converter.models_loaded:
        return jsonify({'error': 'Models not loaded. Please wait for initialization.'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Get processing parameters
    language = request.form.get('language', 'si')
    dpi = int(request.form.get('dpi', 300))
    confidence = float(request.form.get('confidence', 75.0))
    save_json = request.form.get('save_json', 'true').lower() == 'true'
    save_images = request.form.get('save_images', 'false').lower() == 'true'
    
    # Generate unique task ID
    task_id = f"task_{int(time.time() * 1000)}"
    progress_queues[task_id] = queue.Queue(maxsize=PROGRESS_QUEUE_MAXSIZE)
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(filename)[0]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(file_path)
        
        # Create output directory
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], f"{timestamp}_{base_name}_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Start processing in background thread
        def process_async():
            try:
                # Clean up old progress queues before starting
                cleanup_old_progress_queues()
                
                success, result = process_file(file_path, output_dir, language, dpi, confidence, save_json, save_images, task_id)
                
                if success:
                    # The complete message with word_file is already sent from process_file
                    # This is just a fallback completion message
                    emit_progress(task_id, 'complete', 'Processing completed successfully!', 100)
                else:
                    emit_progress(task_id, 'error', f'Processing failed: {result.get("error", "Unknown error")}', 0)
                    
            except Exception as e:
                emit_progress(task_id, 'error', f'Processing error: {str(e)}', 0)
            finally:
                # Clean up progress queue after a delay
                time.sleep(5)
                if task_id in progress_queues:
                    del progress_queues[task_id]
        
        threading.Thread(target=process_async, daemon=True).start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Processing started'
        })
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        if task_id in progress_queues:
            del progress_queues[task_id]
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/api/progress/<task_id>')
def get_progress(task_id):
    """Stream progress updates for a specific task"""
    def generate():
        if task_id not in progress_queues:
            yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
            return
            
        while True:
            try:
                # Get progress update with timeout
                progress_data = progress_queues[task_id].get(timeout=30)
                yield f"data: {json.dumps(progress_data)}\n\n"
                
                # If this is a final status, break
                if progress_data.get('type') in ['complete', 'error']:
                    break
                    
            except queue.Empty:
                # Send keepalive
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

def process_file(input_path, output_dir, language, dpi, confidence, save_json, save_images, task_id):
    """Process the uploaded file with detailed progress updates - one page at a time with memory cleanup"""
    log_messages = []
    
    try:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        file_ext = os.path.splitext(input_path)[1].lower()
        
        # Get initial memory usage
        initial_memory = get_memory_usage()
        emit_progress(task_id, 'file_analysis', f"Analyzing file: {base_name}", 5, memory_info=initial_memory)
        
        # Step 1: Analyze file and get page count
        if file_ext == '.pdf':
            emit_progress(task_id, 'pdf_analysis', "Analyzing PDF structure...", 10)
            log_messages.append("Analyzing PDF structure...")
            total_pages = convert_pdf_to_images(input_path, dpi)
            log_messages.append(f"PDF has {total_pages} pages")
            emit_progress(task_id, 'pdf_analysis', f"PDF has {total_pages} pages", 20)
            is_pdf = True
        else:
            total_pages = 1
            log_messages.append(f"Single image file: {input_path}")
            emit_progress(task_id, 'file_analysis', f"Single image file: {base_name}", 20)
            is_pdf = False
        
        all_ocr_results = []
        all_layout_results = []
        
        emit_progress(task_id, 'processing_start', f"Starting processing of {total_pages} page(s)", 25)
        
        # Step 2: Process each page individually with memory cleanup
        for page_num in range(1, total_pages + 1):
            page_progress = 25 + ((page_num - 1) / total_pages) * 70  # 25% to 95%
            
            # Load page image
            if is_pdf:
                emit_progress(task_id, 'page_loading', f"Loading page {page_num}/{total_pages}", page_progress - 2, page_num, total_pages)
                image = get_pdf_page_image(input_path, page_num - 1, dpi)  # PyMuPDF uses 0-based indexing
            else:
                image = Image.open(input_path)
            
            # Get memory usage before processing page
            pre_page_memory = get_memory_usage()
            emit_progress(task_id, 'page_start', f"Processing page {page_num}/{total_pages}", page_progress, page_num, total_pages, memory_info=pre_page_memory)
            log_messages.append(f"Processing page {page_num}/{total_pages}")
            
            try:
                # OCR for this page
                log_messages.append(f"Performing OCR on page {page_num}...")
                ocr_result = converter.ocr_workflow(image, [language], page_num, task_id)
                
                # Ensure OCR result has proper structure
                if not isinstance(ocr_result, dict):
                    ocr_result = {
                        'page_number': page_num,
                        'page_width': image.size[0] if image else 0,
                        'page_height': image.size[1] if image else 0,
                        'text_lines': [],
                        'error': 'Invalid OCR result format'
                    }
                all_ocr_results.append(ocr_result)
                
                # Layout analysis for this page
                log_messages.append(f"Performing layout analysis on page {page_num}...")
                layout_result = converter.yolo_layout_analysis_workflow(image, page_num, task_id)
                
                # Ensure layout result has proper structure
                if not isinstance(layout_result, list):
                    layout_result = [{'bboxes': [], 'image_bbox': None, 'error': 'Invalid layout result format'}]
                
                all_layout_results.append({
                    'page_number': page_num,
                    'layout_predictions': layout_result
                })
                
                # Save intermediate files if requested
                if save_json:
                    emit_progress(task_id, 'saving_intermediate', f"Saving intermediate files for page {page_num}", page_progress + 5, page_num, total_pages)
                    
                    # Save OCR results
                    ocr_file = os.path.join(output_dir, f"page_{page_num:03d}_ocr.json")
                    with open(ocr_file, 'w', encoding='utf-8') as f:
                        json.dump(ocr_result, f, indent=2, ensure_ascii=False)
                    
                    # Save layout results
                    layout_file = os.path.join(output_dir, f"page_{page_num:03d}_layout.json")
                    with open(layout_file, 'w', encoding='utf-8') as f:
                        json.dump(layout_result, f, indent=2, ensure_ascii=False)
                
                # Save processed images with bounding boxes if requested
                if save_images:
                    try:
                        emit_progress(task_id, 'saving_images', f"Saving processed image for page {page_num}", page_progress + 8, page_num, total_pages)
                        
                        # Create image with OCR bounding boxes
                        img_with_boxes = image.copy()
                        draw = ImageDraw.Draw(img_with_boxes)
                        
                        # Draw OCR bounding boxes
                        if 'text_lines' in ocr_result:
                            for line in ocr_result['text_lines']:
                                if line.get('bbox'):
                                    draw.rectangle(line['bbox'], outline=(255, 0, 0), width=2)

                        # Check if layout_result is a list and has valid data
                        if isinstance(layout_result, list) and len(layout_result) > 0 and 'bboxes' in layout_result[0]:
                            for layout in layout_result[0]['bboxes']:
                                if layout.get('bbox'):
                                    draw.rectangle(layout['bbox'], outline=(0, 255, 0), width=4)
                        
                        # Save processed image
                        img_file = os.path.join(output_dir, f"page_{page_num:03d}_processed.png")
                        img_with_boxes.save(img_file)
                        
                    except Exception as e:
                        log_messages.append(f"Could not save processed image for page {page_num}: {e}")
                        emit_progress(task_id, 'warning', f"Could not save processed image for page {page_num}: {e}", page_progress + 8, page_num, total_pages)
                
                # Clean up memory after processing this page
                converter.cleanup_after_page()
                
                # Get memory usage after cleanup
                post_cleanup_memory = get_memory_usage()
                emit_progress(task_id, 'page_complete', f"Completed page {page_num}/{total_pages}", page_progress + 10, page_num, total_pages, memory_info=post_cleanup_memory)
                
                # Clear the image from memory
                del image
                
            except Exception as e:
                log_messages.append(f"Error processing page {page_num}: {e}")
                emit_progress(task_id, 'page_error', f"Error processing page {page_num}: {str(e)}", page_progress, page_num, total_pages)
                
                # Add empty results for failed page to maintain list structure
                all_ocr_results.append({
                    'page_number': page_num,
                    'page_width': image.size[0] if image else 0,
                    'page_height': image.size[1] if image else 0,
                    'text_lines': [],
                    'error': str(e)
                })
                
                all_layout_results.append({
                    'page_number': page_num,
                    'layout_predictions': [{'bboxes': [], 'image_bbox': None, 'error': str(e)}]
                })
                
                # Continue with next page instead of failing completely
                continue
        
        # Step 3: Generate Word document
        emit_progress(task_id, 'word_generation', "Generating Word document...", 95)
        log_messages.append("Generating Word document...")
        
        doc_path = os.path.join(output_dir, f"{base_name}_converted.docx")
        # For Word document creation, we need a sample image for dimensions
        sample_image = None
        if is_pdf and total_pages > 0:
            sample_image = get_pdf_page_image(input_path, 0, dpi)  # Use first page as sample
        elif not is_pdf:
            sample_image = Image.open(input_path)
        
        success = create_word_document(sample_image, all_ocr_results, all_layout_results, doc_path, confidence)
        
        if success:
            log_messages.append(f"Document saved to: {doc_path}")
            filename = os.path.basename(doc_path)
            logger.info(f"Word document created: {filename}")
            
            # Final memory cleanup
            cleanup_memory()
            final_memory = get_memory_usage()
            
            # Send completion with word_file information
            emit_progress(task_id, 'complete', f"Document saved: {filename}", 100, word_file=filename, memory_info=final_memory)
            return True, {
                'word_file': filename,  # Return just the filename
                'log': log_messages
            }
        else:
            log_messages.append("Failed to create Word document")
            emit_progress(task_id, 'error', "Failed to create Word document", 0)
            return False, {
                'error': 'Failed to create Word document',
                'log': log_messages
            }
            
    except Exception as e:
        log_messages.append(f"Error during processing: {e}")
        emit_progress(task_id, 'error', f"Error during processing: {e}", 0)
        return False, {
            'error': str(e),
            'log': log_messages
        }

def convert_pdf_to_images(pdf_path, dpi):
    """Convert PDF to a list of PIL Image objects using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        logger.info(f"PDF has {total_pages} pages")
        doc.close()
        return total_pages
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        raise

def get_pdf_page_image(pdf_path, page_num, dpi):
    """Get a single page image from PDF"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        # Calculate zoom factor based on DPI (72 DPI is the default)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception as e:
        logger.error(f"Error converting PDF page {page_num} to image: {e}")
        raise

def create_word_document(image, ocr_results, layout_results, output_path, confidence):
    """Create Word document from OCR and layout results"""
    try:
        # Create Word document processor with current confidence threshold
        word_processor = WordDocumentProcessor(confidence)
        return word_processor.create_word_document(image, ocr_results, layout_results, output_path)
    except Exception as e:
        logger.error(f"Error creating Word document: {e}")
        return False



@app.route('/api/download/<path:filename>')
def download_file(filename):
    """Download processed files"""
    try:
        logger.info(f"Download request for: {filename}")
        
        # Search for the file in all subdirectories of the output folder
        output_folder = app.config['OUTPUT_FOLDER']
        file_path = None
        
        # First try direct path
        direct_path = os.path.join(output_folder, filename)
        logger.info(f"Checking direct path: {direct_path}")
        if os.path.exists(direct_path):
            file_path = direct_path
            logger.info(f"File found at direct path: {file_path}")
        else:
            # Search in subdirectories
            logger.info(f"Searching subdirectories in: {output_folder}")
            for root, dirs, files in os.walk(output_folder):
                logger.info(f"Checking directory: {root}, files: {files}")
                if filename in files:
                    file_path = os.path.join(root, filename)
                    logger.info(f"File found in subdirectory: {file_path}")
                    break
        
        if file_path and os.path.exists(file_path):
            logger.info(f"Sending file: {file_path}")
            return send_file(file_path, as_attachment=True)
        else:
            logger.error(f"File not found: {filename}")
            logger.error(f"Searched in: {output_folder}")
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': f'Download error: {str(e)}'}), 500

if __name__ == '__main__':
    # Set environment variables for performance
    os.environ["RECOGNITION_BATCH_SIZE"] = "256"
    os.environ["DETECTOR_BATCH_SIZE"] = "18"
    os.environ["ORDER_BATCH_SIZE"] = "16"
    os.environ["RECOGNITION_STATIC_CACHE"] = "true"
    
    # Configure TorchDynamo
    torch._dynamo.config.capture_scalar_outputs = True
    
    # Set PyTorch memory management
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
        torch.backends.cudnn.benchmark = True
    
    # Initialize models on startup
    initialize_converter()
    
    app.run(debug=True, host='0.0.0.0', port=5001) 