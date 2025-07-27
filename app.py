import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
import threading
import gc
from PIL import Image, ImageDraw
import torch
import logging
import re
from pdf2image import convert_from_path
from ultralytics import YOLO

# Import Surya models
from intellidoc.ocr import run_ocr
from intellidoc.detection import batch_text_detection
from intellidoc.layout import batch_layout_detection
from intellidoc.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from intellidoc.model.recognition.model import load_model as load_rec_model
from intellidoc.model.recognition.processor import load_processor as load_rec_processor
from intellidoc.settings import settings
# Import Word document processor
from word_processor import WordDocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Image.Image):
            return "Image object (not serializable)"
        if hasattr(obj, '__dict__'):
            return {k: self.default(v) for k, v in obj.__dict__.items()}
        return str(obj)

def serialize_result(result):
    return json.dumps(result, cls=CustomJSONEncoder, indent=2)

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
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "mps"
        self.models_loaded = False
        self.det_model = None
        self.det_processor = None
        self.rec_model = None
        self.rec_processor = None
        self.layout_model = None
        self.layout_processor = None
        self.order_model = None
        self.yolo_model_path = yolo_model_path
        # self.order_processor = None
        
    def load_models(self):
        """Load all required models"""
        try:
            logger.info("Loading detection models...")
            self.det_processor, self.det_model = load_det_processor(), load_det_model()
            
            logger.info("Loading recognition models...")
            # Set static cache to True to avoid TorchDynamo compilation issues
            settings.RECOGNITION_STATIC_CACHE = True
            self.rec_model, self.rec_processor = load_rec_model(), load_rec_processor()
            
            # logger.info("Loading layout models...")
            # self.layout_model = load_det_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
            # self.layout_processor = load_det_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)

            logger.info("Loading YOLO layout models...")
            self.yolo_model = YOLO(self.yolo_model_path)
            self.yolo_model.to(self.device)
            
            # Disable model compilation for recognition model to avoid TorchDynamo issues
            # The static cache setting is already enabled above
            logger.info("Recognition model loaded without compilation to avoid TorchDynamo issues")
            
            self.models_loaded = True
            logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def ocr_workflow(self, image, langs=['en'], page_num=1):
        """Perform OCR on a single page and return results with bounding boxes"""
        if not self.models_loaded:
            raise Exception("Models not loaded")
            
        try:
            logger.info(f"Starting OCR workflow for page {page_num}")
            predictions = run_ocr([image], [langs], self.det_model, self.det_processor, 
                                self.rec_model, self.rec_processor)
            
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
            
            logger.info(f"OCR completed for page {page_num}")
            return page_result
            
        except Exception as e:
            logger.error(f"Error during OCR workflow: {e}")
            return {'page_number': page_num, 'error': str(e)}
    
    # def layout_analysis_workflow(self, image, page_num=1):
    #     """Perform layout analysis on a single page"""
    #     if not self.models_loaded:
    #         raise Exception("Models not loaded")
            
    #     try:
    #         logger.info(f"Starting layout analysis for page {page_num}")
            
    #         # Text detection
    #         line_predictions = batch_text_detection([image], self.det_model, self.det_processor)
            
    #         # Layout detection
    #         layout_predictions = batch_layout_detection([image], self.layout_model, 
    #                                                   self.layout_processor, line_predictions)
            
    #         # Convert to serializable format
    #         serializable_predictions = []
    #         for pred in layout_predictions:
    #             serializable_pred = {
    #                 'bboxes': [
    #                     {
    #                         'bbox': bbox.bbox.tolist() if hasattr(bbox.bbox, 'tolist') and bbox.bbox is not None else bbox.bbox,
    #                         'polygon': bbox.polygon.tolist() if hasattr(bbox.polygon, 'tolist') and bbox.polygon is not None else bbox.polygon,
    #                         'confidence': bbox.confidence,
    #                         'label': bbox.label
    #                     }
    #                     for bbox in pred.bboxes
    #                 ],
    #                 'image_bbox': pred.image_bbox.tolist() if hasattr(pred.image_bbox, 'tolist') and pred.image_bbox is not None else pred.image_bbox
    #             }
    #             serializable_predictions.append(serializable_pred)
            
    #         logger.info(f"Layout analysis completed for page {page_num}")
    #         return serializable_predictions
            
    #     except Exception as e:
    #         logger.error(f"Error during layout analysis: {e}")
    #         return {'error': str(e)}

    def yolo_layout_analysis_workflow(self, image, page_num=1):
        """Perform layout analysis on a single page"""
        if not self.models_loaded:
            raise Exception("Models not loaded")
            
        try:
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
            
            # YOLO prediction with optimized parameters to avoid NMS timeout
            results = self.yolo_model.predict(
                image, 
                save=False, 
                conf=0.7,  # Higher confidence threshold
                iou=0.5,   # IOU threshold for NMS
                max_det=50,  # Limit maximum detections
                verbose=False  # Reduce logging
            )
            
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
            
            logger.info(f"Layout analysis completed for page {page_num}")
            return serializable_predictions
            
        except Exception as e:
            logger.error(f"Error during layout analysis: {e}")
            return {'error': str(e)}
    
    def _convert_xyxy_to_polygon(self, xyxy):
        """Convert xyxy format [x1, y1, x2, y2] to polygon format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]"""
        if xyxy is None or len(xyxy) != 4:
            return None
        x1, y1, x2, y2 = xyxy
        return [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]

class PDFWordConverterGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PDF/Image to Word Converter with Layout Analysis")
        self.root.geometry("800x600")
        self.converter = PDFWordConverter(yolo_model_path="weights/best.pt")
        self.setup_ui()
        self.load_models_async()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Title
        title_label = ttk.Label(main_frame, text="PDF/Image to Word Converter with Layout Analysis", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection
        ttk.Label(main_frame, text="Select File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(main_frame, textvariable=self.file_path_var, width=60)
        self.file_entry.grid(row=1, column=1, padx=(10, 0), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_file).grid(row=1, column=2, padx=(10, 0), pady=5)
        
        # Output directory selection
        ttk.Label(main_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar()
        self.output_entry = ttk.Entry(main_frame, textvariable=self.output_dir_var, width=60)
        self.output_entry.grid(row=2, column=1, padx=(10, 0), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output_dir).grid(row=2, column=2, padx=(10, 0), pady=5)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="5")
        settings_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)
        
        # Language selection
        ttk.Label(settings_frame, text="Language:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.lang_var = tk.StringVar(value="si")
        lang_combo = ttk.Combobox(settings_frame, textvariable=self.lang_var, 
                                 values=["si","en", "es", "fr", "de", "pt", "it"], width=10)
        lang_combo.grid(row=0, column=1, padx=5)
        
        # DPI setting for PDF
        ttk.Label(settings_frame, text="PDF DPI:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.dpi_var = tk.IntVar(value=300)  # Reduced from 300 to 200 for faster processing
        ttk.Spinbox(settings_frame, from_=150, to=600, increment=50, 
                   textvariable=self.dpi_var, width=10).grid(row=0, column=3, padx=5)
        
        # OCR Confidence threshold
        ttk.Label(settings_frame, text="OCR Confidence (%):").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.confidence_var = tk.DoubleVar(value=75.0)
        ttk.Spinbox(settings_frame, from_=50.0, to=100.0, increment=5.0, 
                   textvariable=self.confidence_var, width=10).grid(row=1, column=1, padx=5)
        
        # Processing options
        self.save_json_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Save JSON intermediate files", 
                       variable=self.save_json_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        self.save_images_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Save processed images with bounding boxes", 
                       variable=self.save_images_var).grid(row=2, column=2, columnspan=2, sticky=tk.W, pady=5)
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="Process Document", 
                                       command=self.process_document, state=tk.DISABLED)
        self.process_button.grid(row=4, column=1, pady=20)
        
        # Progress section
        self.progress_var = tk.StringVar()
        self.progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=5, column=0, columnspan=3, pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.grid(row=6, column=0, columnspan=3, sticky="ew", pady=5)
        
        # Status
        self.status_var = tk.StringVar()
        self.status_var.set("Loading models...")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.grid(row=7, column=0, columnspan=3, pady=10)
        
        # Log text area
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="5")
        log_frame.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=10)
        
        self.log_text = tk.Text(log_frame, height=10, width=80)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(8, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
    
    def load_models_async(self):
        def load():
            try:
                success = self.converter.load_models()
                self.root.after(0, lambda: self.models_loaded(success))
            except Exception as e:
                self.root.after(0, lambda: self.models_failed(str(e)))
        
        threading.Thread(target=load, daemon=True).start()
    
    def models_loaded(self, success):
        if success:
            self.status_var.set("Models loaded successfully! Ready to process documents.")
            self.process_button.config(state=tk.NORMAL)
            self.log_message("Models loaded successfully!")
        else:
            self.status_var.set("Failed to load models.")
            self.log_message("Failed to load models. Please check console for errors.")
    
    def models_failed(self, error_msg):
        self.status_var.set(f"Failed to load models: {error_msg}")
        self.log_message(f"Failed to load models: {error_msg}")
        messagebox.showerror("Error", f"Failed to load models: {error_msg}")
    
    def browse_file(self):
        filetypes = [
            ("Supported files", "*.pdf *.jpg *.jpeg *.png *.bmp *.tiff"),
            ("PDF files", "*.pdf"),
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
        file_path = filedialog.askopenfilename(title="Select PDF or Image file", filetypes=filetypes)
        if file_path:
            self.file_path_var.set(file_path)
            # Set default output directory
            if not self.output_dir_var.get():
                self.output_dir_var.set(os.path.dirname(file_path))
    
    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Select output directory")
        if directory:
            self.output_dir_var.set(directory)
    
    def log_message(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_progress(self, value, message=""):
        self.progress_bar['value'] = value
        if message:
            self.progress_var.set(message)
        self.root.update_idletasks()
    
    def process_document(self):
        input_path = self.file_path_var.get()
        output_dir = self.output_dir_var.get()
        
        if not input_path or not output_dir:
            messagebox.showerror("Error", "Please select both input file and output directory.")
            return
        
        if not os.path.exists(input_path):
            messagebox.showerror("Error", "Input file does not exist.")
            return
        
        if not os.path.exists(output_dir):
            messagebox.showerror("Error", "Output directory does not exist.")
            return
        
        self.process_button.config(state=tk.DISABLED)
        self.progress_bar['value'] = 0
        
        def process_thread():
            try:
                success = self.process_file(input_path, output_dir)
                self.root.after(0, lambda: self.processing_complete(success))
            except Exception as e:
                self.root.after(0, lambda: self.processing_failed(str(e)))
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def process_file(self, input_path, output_dir):
        """Main processing function"""
        try:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            file_ext = os.path.splitext(input_path)[1].lower()
            
            # Create output subdirectory
            output_subdir = os.path.join(output_dir, f"{base_name}_output")
            os.makedirs(output_subdir, exist_ok=True)
            
            # Step 1: Convert to images if PDF
            if file_ext == '.pdf':
                self.log_message("Converting PDF to images...")
                images = convert_from_path(input_path, dpi=self.dpi_var.get())
                self.log_message(f"PDF converted to {len(images)} pages")
            else:
                images = [Image.open(input_path)]
                self.log_message(f"Loaded single image: {input_path}")
            
            total_pages = len(images)
            all_ocr_results = []
            all_layout_results = []
            
            # Step 2: Process each page
            for i, image in enumerate(images):
                page_num = i + 1
                self.update_progress((i / total_pages) * 50, f"Processing page {page_num}/{total_pages}")
                
                # OCR for this page
                self.log_message(f"Performing OCR on page {page_num}...")
                ocr_result = self.converter.ocr_workflow(image, [self.lang_var.get()], page_num)
                all_ocr_results.append(ocr_result)
                
                # Layout analysis for this page
                self.log_message(f"Performing layout analysis on page {page_num}...")
                #layout_result = self.converter.layout_analysis_workflow(image, page_num)
                layout_result = self.converter.yolo_layout_analysis_workflow(image, page_num)

                all_layout_results.append({
                    'page_number': page_num,
                    'layout_predictions': layout_result
                })
                
                # Save intermediate files if requested
                if self.save_json_var.get():
                    # Save OCR results
                    ocr_file = os.path.join(output_subdir, f"page_{page_num:03d}_ocr.json")
                    with open(ocr_file, 'w', encoding='utf-8') as f:
                        json.dump(ocr_result, f, indent=2, ensure_ascii=False)
                    
                    # Save layout results
                    layout_file = os.path.join(output_subdir, f"page_{page_num:03d}_layout.json")
                    with open(layout_file, 'w', encoding='utf-8') as f:
                        json.dump(layout_result, f, indent=2, ensure_ascii=False)
                
                # Save processed images with bounding boxes if requested
                if self.save_images_var.get():
                    try:
                        # Create image with OCR bounding boxes
                        img_with_boxes = image.copy()
                        draw = ImageDraw.Draw(img_with_boxes)
                        
                        # Draw OCR bounding boxes
                        if 'text_lines' in ocr_result:
                            for line in ocr_result['text_lines']:
                                if line.get('bbox'):
                                    draw.rectangle(line['bbox'], outline=(255, 0, 0), width=2)

                        for layout in layout_result[0]['bboxes']:
                            if layout.get('bbox'):
                                draw.rectangle(layout['bbox'], outline=(0, 255, 0), width=4)
                        
                        # Save processed image
                        img_file = os.path.join(output_subdir, f"page_{page_num:03d}_processed.png")
                        img_with_boxes.save(img_file)
                        
                    except Exception as e:
                        self.log_message(f"Could not save processed image for page {page_num}: {e}")
            
            # Step 3: Generate Word document
            self.update_progress(75, "Generating Word document...")
            self.log_message("Generating Word document...")
            
            doc_path = os.path.join(output_subdir, f"{base_name}_converted.docx")
            success = self.create_word_document(image, all_ocr_results, all_layout_results, doc_path)
            
            if success:
                self.update_progress(100, "Processing completed successfully!")
                self.log_message(f"Document saved to: {doc_path}")
                return True
            else:
                self.log_message("Failed to create Word document")
                return False
                
        except Exception as e:
            self.log_message(f"Error during processing: {e}")
            return False
    
    def create_word_document(self, image, ocr_results, layout_results, output_path):
        """Create Word document from OCR and layout results"""
        try:
            # Create Word document processor with current confidence threshold
            word_processor = WordDocumentProcessor(self.confidence_var.get())
            return word_processor.create_word_document(image, ocr_results, layout_results, output_path)
        except Exception as e:
            self.log_message(f"Error creating Word document: {e}")
            return False
    
    def processing_complete(self, success):
        self.process_button.config(state=tk.NORMAL)
        
        if success:
            self.log_message("Processing completed successfully!")
            messagebox.showinfo("Success", "Document processing completed successfully!")
        else:
            self.log_message("Processing failed. Please check the log for details.")
            messagebox.showerror("Error", "Processing failed. Please check the log for details.")
    
    def processing_failed(self, error_msg):
        self.process_button.config(state=tk.NORMAL)
        self.log_message(f"Processing failed: {error_msg}")
        messagebox.showerror("Error", f"Processing failed: {error_msg}")
    
    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nGUI closed by user.")
        finally:
            # Cleanup
            gc.collect()


if __name__ == "__main__":
    # Set environment variables for performance
    os.environ["RECOGNITION_BATCH_SIZE"] = "256"
    os.environ["DETECTOR_BATCH_SIZE"] = "18"
    os.environ["ORDER_BATCH_SIZE"] = "16"
    os.environ["RECOGNITION_STATIC_CACHE"] = "true"
    
    # Configure TorchDynamo
    torch._dynamo.config.capture_scalar_outputs = True
    
    app = PDFWordConverterGUI()
    app.run()