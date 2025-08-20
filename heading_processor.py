import logging
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter
import statistics

logger = logging.getLogger(__name__)

class HeadingProcessor:
    def __init__(self, ocr_results: List[Dict], layout_results: List[Dict]):
        """
        Initialize the heading processor with OCR and layout results.
        
        Args:
            ocr_results: List of OCR results for each page
            layout_results: List of layout results for each page
        """
        self.ocr_results = ocr_results
        self.layout_results = layout_results
        
    def __call__(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Process headings and return updated OCR and layout results.
        
        Returns:
            Tuple of (updated_ocr_results, updated_layout_results)
        """
        try:
            # Process each page
            for page_idx, (ocr_result, layout_result) in enumerate(zip(self.ocr_results, self.layout_results)):
                logger.info(f"Processing headings for page {page_idx + 1}")
                
                # Extract text lines and their properties
                text_lines = ocr_result.get('text_lines', [])
                if not text_lines:
                    continue
                
                # Filter layout boxes to only include Text and SectionHeader labels
                filtered_layout_boxes = self._filter_layout_boxes(layout_result)
                
                # Match text lines with layout boxes
                matched_text_lines = self._match_text_with_layout(text_lines, filtered_layout_boxes)
                
                # Analyze text sizes and identify headings
                heading_analysis = self._analyze_text_sizes(matched_text_lines)
                
                # Identify headings based on size and spacing
                heading_lines = self._identify_headings(matched_text_lines, heading_analysis)
                
                # Categorize headings
                categorized_headings = self._categorize_headings(heading_lines, heading_analysis)
                
                # Update layout results with heading information
                updated_layout_result = self._update_layout_with_headings(
                    layout_result, categorized_headings, matched_text_lines
                )
                
                # Update OCR results with heading information
                updated_ocr_result = self._update_ocr_with_headings(
                    ocr_result, categorized_headings
                )
                
                # Update the results
                self.ocr_results[page_idx] = updated_ocr_result
                self.layout_results[page_idx] = updated_layout_result
                
            return self.ocr_results, self.layout_results
            
        except Exception as e:
            logger.error(f"Error in heading processing: {e}")
            return self.ocr_results, self.layout_results
    
    def _filter_layout_boxes(self, layout_result: Dict) -> List[Dict]:
        """
        Filter layout boxes to only include Text and SectionHeader labels.
        
        Args:
            layout_result: Layout result dictionary
            
        Returns:
            List of filtered layout boxes
        """
        filtered_boxes = []
        
        # Handle different layout result structures
        if isinstance(layout_result, list) and len(layout_result) > 0:
            # If layout_result is a list, take the first element
            layout_data = layout_result[0]
        else:
            layout_data = layout_result
        
        bboxes = layout_data.get('bboxes', [])
        
        for bbox in bboxes:
            label = bbox.get('label', '')
            # Only include Text and SectionHeader labels
            if label in ['Text', 'SectionHeader']:
                filtered_boxes.append(bbox)
        
        return filtered_boxes
    
    def _match_text_with_layout(self, text_lines: List[Dict], layout_boxes: List[Dict]) -> List[Dict]:
        """
        Match text lines with layout boxes and add layout information.
        
        Args:
            text_lines: List of text line dictionaries
            layout_boxes: List of filtered layout boxes
            
        Returns:
            List of text lines with layout information
        """
        matched_lines = []
        
        for text_line in text_lines:
            text_bbox = text_line.get('bbox')
            if not text_bbox or len(text_bbox) != 4:
                continue
            
            # Find the layout box that contains this text line
            matched_layout = None
            for layout_box in layout_boxes:
                layout_bbox = layout_box.get('bbox')
                if not layout_bbox or len(layout_bbox) != 4:
                    continue
                
                # Check if text line is contained within layout box
                if (text_bbox[0] >= layout_bbox[0] and text_bbox[1] >= layout_bbox[1] and
                    text_bbox[2] <= layout_bbox[2] and text_bbox[3] <= layout_bbox[3]):
                    matched_layout = layout_box
                    break
            
            # Create enhanced text line with layout information
            enhanced_line = text_line.copy()
            if matched_layout:
                enhanced_line['layout_label'] = matched_layout.get('label', '')
                enhanced_line['layout_confidence'] = matched_layout.get('confidence', 0)
            
            matched_lines.append(enhanced_line)
        
        return matched_lines
    
    def _analyze_text_sizes(self, text_lines: List[Dict]) -> Dict[str, Any]:
        """
        Analyze text sizes and calculate statistics.
        Calculate font sizes from bbox heights since font_size is not available.
        
        Args:
            text_lines: List of text line dictionaries
            
        Returns:
            Dictionary containing size analysis
        """
        # Extract font sizes (calculated from heights) and heights
        font_sizes = []
        heights = []
        line_data = []
        
        for line in text_lines:
            # Calculate height from bbox
            bbox = line.get('bbox')
            if bbox and len(bbox) == 4:
                height = bbox[3] - bbox[1]
                heights.append(height)
                
                # Calculate font size from height (approximate)
                # Font size is typically about 70-80% of the line height
                font_size = height * 0.75
                font_sizes.append(font_size)
            
            # Store line data for analysis
            line_data.append({
                'text': line.get('text', ''),
                'bbox': bbox,
                'font_size': font_size if bbox and len(bbox) == 4 else None,
                'height': height if bbox and len(bbox) == 4 else None,
                'confidence': line.get('confidence', 0),
                'layout_label': line.get('layout_label', ''),
                'layout_confidence': line.get('layout_confidence', 0)
            })
        
        # Calculate statistics
        analysis = {
            'line_data': line_data,
            'font_sizes': font_sizes,
            'heights': heights,
            'avg_font_size': statistics.mean(font_sizes) if font_sizes else None,
            'avg_height': statistics.mean(heights) if heights else None,
            'font_size_std': statistics.stdev(font_sizes) if len(font_sizes) > 1 else 0,
            'height_std': statistics.stdev(heights) if len(heights) > 1 else 0,
            'font_size_frequencies': Counter(font_sizes) if font_sizes else Counter(),
            'height_frequencies': Counter(heights) if heights else Counter()
        }
        
        return analysis
    
    def _identify_headings(self, text_lines: List[Dict], analysis: Dict[str, Any]) -> List[Dict]:
        """
        Identify heading lines based on size and spacing criteria.
        Only consider Text and SectionHeader labels.
        
        Args:
            text_lines: List of text line dictionaries
            analysis: Size analysis results
            
        Returns:
            List of heading line dictionaries
        """
        heading_lines = []
        avg_font_size = analysis['avg_font_size']
        avg_height = analysis['avg_height']
        
        # Define thresholds for heading detection
        size_threshold = 1.2  # 20% larger than average
        spacing_threshold = 10  # Minimum pixels of spacing
        
        for line in analysis['line_data']:
            text = line['text'].strip()
            if not text:
                continue
            
            # Skip very short text (likely not headings)
            if len(text) < 3:
                continue
            
            # Check if it's a single line (no line breaks)
            if '\n' in text or '\r' in text:
                continue
            
            # Only consider Text and SectionHeader labels
            layout_label = line.get('layout_label', '')
            if layout_label not in ['Text', 'SectionHeader']:
                continue
            
            is_heading = False
            heading_score = 0
            
            # Check font size criteria
            if line['font_size'] and avg_font_size:
                size_ratio = line['font_size'] / avg_font_size
                if size_ratio > size_threshold:
                    heading_score += 2
                    is_heading = True
            
            # Check height criteria
            if line['height'] and avg_height:
                height_ratio = line['height'] / avg_height
                if height_ratio > size_threshold:
                    heading_score += 1
                    is_heading = True
            
            # Check spacing (top and bottom margins)
            if line['bbox']:
                spacing_score = self._check_spacing(line['bbox'], text_lines)
                if spacing_score > spacing_threshold:
                    heading_score += 1
            
            # Check if text looks like a heading (capitalization, length)
            if self._looks_like_heading(text):
                heading_score += 1
            
            # Final decision
            if heading_score >= 2 or is_heading:
                heading_lines.append({
                    **line,
                    'heading_score': heading_score,
                    'is_heading': True
                })
        
        return heading_lines
    
    def _check_spacing(self, bbox: List[float], text_lines: List[Dict]) -> float:
        """
        Check the spacing around a text line.
        
        Args:
            bbox: Bounding box of the text line
            text_lines: All text lines for comparison
            
        Returns:
            Spacing score
        """
        if not bbox or len(bbox) != 4:
            return 0
        
        x1, y1, x2, y2 = bbox
        line_height = y2 - y1
        
        # Calculate spacing above and below
        spacing_above = float('inf')
        spacing_below = float('inf')
        
        for other_line in text_lines:
            other_bbox = other_line.get('bbox')
            if not other_bbox or len(other_bbox) != 4:
                continue
            
            ox1, oy1, ox2, oy2 = other_bbox
            
            # Check if lines overlap horizontally
            if not (x2 < ox1 or ox2 < x1):
                # Line above
                if oy2 < y1:
                    spacing_above = min(spacing_above, y1 - oy2)
                # Line below
                elif oy1 > y2:
                    spacing_below = min(spacing_below, oy1 - y2)
        
        # Return minimum spacing (infinity becomes 0)
        return min(spacing_above if spacing_above != float('inf') else 0,
                  spacing_below if spacing_below != float('inf') else 0)
    
    def _looks_like_heading(self, text: str) -> bool:
        """
        Check if text looks like a heading based on content.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text looks like a heading
        """
        # Check for common heading patterns
        heading_patterns = [
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^[A-Z][a-z\s]+$',  # Title Case
            r'^\d+\.\s+[A-Z]',   # Numbered headings
            r'^[A-Z]\.\s+[A-Z]', # Lettered headings
            r'^Chapter\s+\d+',   # Chapter headings
            r'^Section\s+\d+',   # Section headings
        ]
        
        import re
        for pattern in heading_patterns:
            if re.match(pattern, text):
                return True
        
        # Check if text is short and meaningful
        words = text.split()
        if 1 <= len(words) <= 8:
            # Check if most words start with capital letters
            capitalized_words = sum(1 for word in words if word and word[0].isupper())
            if capitalized_words >= len(words) * 0.7:
                return True
        
        return False
    
    def _categorize_headings(self, heading_lines: List[Dict], analysis: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """
        Categorize headings based on size and frequency.
        Level 1 (Main): Least frequent, largest headings
        Level 2 (Sub): Second least frequent, second largest
        Level 3 (Section): Third least frequent, third largest or SectionHeader labels in smallest/most frequent
        
        Args:
            heading_lines: List of heading lines
            analysis: Size analysis results
            
        Returns:
            Dictionary of categorized headings
        """
        if not heading_lines:
            return {}
        
        # Group headings by size
        size_groups = defaultdict(list)
        avg_font_size = analysis['avg_font_size']
        avg_height = analysis['avg_height']
        
        for heading in heading_lines:
            # Determine primary size metric
            if heading['font_size'] and avg_font_size:
                size_ratio = heading['font_size'] / avg_font_size
                size_key = f"font_{size_ratio:.2f}"
            elif heading['height'] and avg_height:
                size_ratio = heading['height'] / avg_height
                size_key = f"height_{size_ratio:.2f}"
            else:
                size_key = "unknown"
            
            size_groups[size_key].append(heading)
        
        # Categorize based on frequency and size
        categories = {
            'main_headings': [],
            'sub_headings': [],
            'section_headings': []
        }
        
        # Sort size groups by size ratio (largest first) and then by frequency (least frequent first)
        # This ensures Level 1 = largest and least frequent, Level 3 = smallest and most frequent
        sorted_groups = sorted(size_groups.items(), 
                             key=lambda x: (
                                 float(x[0].split('_')[1]) if '_' in x[0] and x[0].split('_')[1] != 'unknown' else 0,  # Size ratio (largest first)
                                 -len(x[1])  # Frequency (least frequent first, so negative)
                             ),
                             reverse=True)
        
        # Assign categories based on 3-level hierarchy
        # Level 1 (Main): Largest and least frequent
        # Level 2 (Sub): Second largest and second least frequent
        # Level 3 (Section): Third largest and third least frequent, or SectionHeader labels in smallest/most frequent
        
        for i, (size_key, headings) in enumerate(sorted_groups):
            if i == 0:
                # First group: largest size, least frequent -> Main Headings (Level 1)
                categories['main_headings'] = headings
            elif i == 1:
                # Second group: second largest size, second least frequent -> Sub Headings (Level 2)
                categories['sub_headings'] = headings
            else:
                # Remaining groups: smaller sizes, more frequent
                # Check if any headings are SectionHeader labels, if so add to section_headings
                # Otherwise, ignore Text labels that are smallest and most frequent
                for heading in headings:
                    if heading.get('layout_label') == 'SectionHeader':
                        categories['section_headings'].append(heading)
                    # Skip Text labels that are smallest and most frequent (don't add to any category)
        
        return categories
    
    def _update_layout_with_headings(self, layout_result: Dict, categorized_headings: Dict[str, List[Dict]], 
                                   text_lines: List[Dict]) -> Dict:
        """
        Update layout results with heading information.
        Remove existing heading boxes before adding new ones to avoid duplicates.
        
        Args:
            layout_result: Original layout result
            categorized_headings: Categorized headings
            text_lines: All text lines
            
        Returns:
            Updated layout result
        """
        updated_layout = layout_result.copy()
        
        # Handle different layout result structures
        if isinstance(updated_layout, list) and len(updated_layout) > 0:
            layout_data = updated_layout[0]
        else:
            layout_data = updated_layout
        
        # Remove existing heading boxes from bboxes
        if 'bboxes' in layout_data:
            layout_data['bboxes'] = [
                bbox for bbox in layout_data['bboxes'] 
                if not bbox.get('label', '').startswith('Heading_')
            ]
        
        # Remove existing heading predictions
        if 'layout_predictions' in updated_layout:
            updated_layout['layout_predictions'] = [
                pred for pred in updated_layout['layout_predictions'] 
                if pred.get('type') != 'headings'
            ]
        
        # Create new layout boxes for headings
        heading_boxes = []
        
        for category, headings in categorized_headings.items():
            for heading in headings:
                bbox = heading.get('bbox')
                if bbox and len(bbox) == 4:
                    # Create layout box for heading
                    heading_box = {
                        'bbox': bbox,
                        'polygon': heading.get('polygon', [
                            [bbox[0], bbox[1]], [bbox[2], bbox[1]], 
                            [bbox[2], bbox[3]], [bbox[0], bbox[3]]
                        ]),
                        'confidence': heading.get('confidence', 0.9),
                        'label': f'Heading_{category.replace("_", "").title()}',
                        'text': heading.get('text', ''),
                        'heading_category': category,
                        'heading_score': heading.get('heading_score', 0)
                    }
                    heading_boxes.append(heading_box)
        
        # Add heading boxes to main bboxes list
        if heading_boxes:
            if 'bboxes' not in layout_data:
                layout_data['bboxes'] = []
            layout_data['bboxes'].extend(heading_boxes)
        
        # Add heading boxes as a new prediction group
        if 'layout_predictions' not in updated_layout:
            updated_layout['layout_predictions'] = []
        
        if heading_boxes:
            updated_layout['layout_predictions'].append({
                'bboxes': heading_boxes,
                'type': 'headings',
                'confidence': 0.9
            })
        
        return updated_layout
    
    def _update_ocr_with_headings(self, ocr_result: Dict, categorized_headings: Dict[str, List[Dict]]) -> Dict:
        """
        Update OCR results with heading information.
        
        Args:
            ocr_result: Original OCR result
            categorized_headings: Categorized headings
            
        Returns:
            Updated OCR result
        """
        updated_ocr = ocr_result.copy()
        
        # Add heading information to text lines
        text_lines = updated_ocr.get('text_lines', [])
        
        # Create a mapping of heading information
        heading_map = {}
        for category, headings in categorized_headings.items():
            for heading in headings:
                # Use bbox as key for matching
                bbox_key = tuple(heading.get('bbox', []))
                heading_map[bbox_key] = {
                    'is_heading': True,
                    'heading_category': category,
                    'heading_score': heading.get('heading_score', 0)
                }
        
        # Update text lines with heading information
        for line in text_lines:
            bbox = line.get('bbox')
            if bbox and len(bbox) == 4:
                bbox_key = tuple(bbox)
                if bbox_key in heading_map:
                    line.update(heading_map[bbox_key])
                else:
                    line['is_heading'] = False
        
        updated_ocr['text_lines'] = text_lines
        updated_ocr['heading_analysis'] = {
            'total_headings': sum(len(headings) for headings in categorized_headings.values()),
            'categories': {k: len(v) for k, v in categorized_headings.items()}
        }
        
        return updated_ocr
