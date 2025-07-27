import cv2
import numpy as np
from PIL import Image
from typing import List, Dict
import logging

class TableRecognition:
    @staticmethod
    def table_recognition(relevant_lines_full: List[dict], layout_box: Dict, page_data: Dict, debug: bool = False) -> Dict:
        """
        Detect table structure and map text content to cells
        
        Args:
            image: PIL Image of the full page
            relevant_lines_full: List of detected text with bboxes
            layout_box: Bounding box of the table region
            page_data: Page dimensions
            
        Returns:
            Dictionary containing table structure and cell contents
        """
        try:
            image = page_data['image']
            # Step 1: Crop the image to table region
            cropped_image = TableRecognition._crop_table_region(image, layout_box)
            
            # Step 2: Detect cell structure using OpenCV
            cells = TableRecognition._detect_table_cells(cropped_image)
            
            if debug:
                print(f"Detected {len(cells)} cells")
                print(f"Cropped image shape: {cropped_image.shape}")
                for i, cell in enumerate(cells):
                    print(f"Cell {i}: {cell}")
                    
            # If no proper cells detected, create a simple grid based on text distribution
            if len(cells) <= 1:
                structured_table = TableRecognition._create_text_based_grid(relevant_lines_full, layout_box, page_data)
            else:
                # Step 3: Map cell coordinates to full page coordinates
                mapped_cells = TableRecognition._map_cells_to_full_page(
                    cells, layout_box, page_data
                )
                
                # Step 4 & 5: Assign text to cells and score
                table_data = TableRecognition._assign_text_to_cells(
                    mapped_cells, relevant_lines_full
                )
                
                # Step 6: Structure the table data
                structured_table = TableRecognition._structure_table_data(table_data)
            
            return structured_table
            
        except Exception as e:
            logging.error(f"Error in table recognition: {str(e)}")
            return {"error": str(e), "cells": [], "rows": 0, "cols": 0}
    
    @staticmethod
    def _crop_table_region(image: Image.Image, layout_box: Dict) -> np.ndarray:
        """Crop the image to the table region"""
        bbox = layout_box['bbox']  # [x1, y1, x2, y2]
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Crop the region
        cropped = img_array[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        return cropped
    
    @staticmethod
    def _detect_table_cells(cropped_image: np.ndarray) -> List[Dict]:
        """Detect table cells using multiple approaches"""
        # Convert to grayscale if needed
        if len(cropped_image.shape) == 3:
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = cropped_image.copy()
        
        # Try method 1: Line detection
        cells = TableRecognition._detect_cells_by_lines(gray)
        
        # Sort cells by position (top to bottom, left to right)
        if cells:
            cells.sort(key=lambda cell: (cell['y'], cell['x']))
        
        return cells
    
    @staticmethod
    def _create_text_based_grid(text_lines: List[dict], layout_box: Dict, page_data: Dict) -> Dict:
        """Create grid based on text positions when line detection fails"""
        if not text_lines:
            return {"error": "No text lines provided", "rows": 0, "cols": 0, "cells": [], "table_data": []}
        
        # Filter text lines that are within the table region
        table_bbox = layout_box['bbox']
        table_texts = []
        
        for text in text_lines:
            text_bbox = text['bbox']
            # Check if text overlaps with table region
            if (text_bbox[0] < table_bbox[2] and text_bbox[2] > table_bbox[0] and text_bbox[1] < table_bbox[3] and text_bbox[3] > table_bbox[1]):                
                # Convert to table-relative coordinates
                rel_x = text_bbox[0] - table_bbox[0]
                rel_y = text_bbox[1] - table_bbox[1]
                rel_x2 = text_bbox[2] - table_bbox[0]
                rel_y2 = text_bbox[3] - table_bbox[1]

                table_texts.append({
                    'text': text['text'],
                    'x': rel_x, 'y': rel_y,
                    'x2': rel_x2, 'y2': rel_y2,
                    'center_x': (rel_x + rel_x2) / 2,
                    'center_y': (rel_y + rel_y2) / 2
                })
        
        if not table_texts:
            return {"error": "No text found in table region", "rows": 0, "cols": 0, "cells": [], "table_data": []}
        
        # Group texts by approximate rows and columns
        table_texts.sort(key=lambda t: (t['center_y'], t['center_x']))
        
        # Find row breaks (large gaps in Y coordinates)
        row_groups = []
        
        if not table_texts:
            return {"error": "No table texts available", "rows": 0, "cols": 0, "cells": [], "table_data": []}

        same_end_group = []
        not_in_end_group = []

        pixel_threshold = 3
        page_end_to_text_end_list = []

        for text_boxes in table_texts:                
            page_end_to_text_end = page_data['image'].width - text_boxes['x2']
            page_end_to_text_end_list.append(page_end_to_text_end)

        sorted_page_end_to_text_end_list = sorted(page_end_to_text_end_list)

        end_groups = []
        for x in sorted_page_end_to_text_end_list:
            if not end_groups or x - end_groups[-1][-1] > pixel_threshold:
                end_groups.append([x])
            else:
                end_groups[-1].append(x)

        longest = max(end_groups, key=len)
        for text_boxes in table_texts:                
            page_end_to_text_end = page_data['image'].width - text_boxes['x2']
            if page_end_to_text_end in longest:
                same_end_group.append(text_boxes)
            else:
                not_in_end_group.append(text_boxes)

        sorted_end_group = sorted(same_end_group, key=lambda t: (t['center_y'], t['center_x']))

        # Group texts into rows based on Y coordinates
        row_groups = []
        for x,ends in enumerate(sorted_end_group):
            current_row = []
            for i, text_boxes in enumerate(not_in_end_group):
                if x == 0:
                    s_e=sorted_end_group[x]['y']
                else:
                    s_e=sorted_end_group[x]['y']- abs((sorted_end_group[x-1]['y2']-sorted_end_group[x]['y'])/2)

                if x == len(sorted_end_group)-1:
                    e_e=sorted_end_group[x]['y2']
                else:
                    e_e=abs((sorted_end_group[x]['y2']-sorted_end_group[x+1]['y'])/2)+sorted_end_group[x]['y2']
                #print(text_boxes['center_y'], s_e, e_e)
                if (text_boxes['center_y'] > s_e and text_boxes['center_y'] < e_e):
                    current_row.append(text_boxes)
            current_row.append(ends)
            row_groups.append(current_row)
            
        
        # Create structured table data directly
        table_structure = {
            "rows": len(row_groups),
            "cols": 2,
            "cells": [],
            "table_data": []
        }
        
        # Process each row to create table structure
        for row_idx, row_texts in enumerate(row_groups):
            # Sort by X coordinate
            row_texts.sort(key=lambda t: t['center_x'])
            row_data = []
            
            start_cell_bbox=[]
            text_in_cell=""

            cell_info={}
            end_text=""
            lswap=False
            for col_idx, text in enumerate(row_texts):

                if col_idx == len(row_texts)-1:
                # Create cell info with full page coordinates
                    cell_bbox = [
                        text['x'] + table_bbox[0],  # Convert back to full page coordinates
                        text['y'] + table_bbox[1],
                        text['x2'] + table_bbox[0],
                        text['y2'] + table_bbox[1]
                    ]
                    
                    cell_info = {
                        'row': row_idx,
                        'col': 1,
                        'text': text['text'],
                        'bbox': cell_bbox,
                        'confidence': 0.8  # Default confidence for text-based grid
                    }

                    end_text=text['text']
                else:

                    start_cell_bbox = [
                        text['x'] + table_bbox[0],  # Convert back to full page coordinates
                        text['y'] + table_bbox[1],
                        text['x2'] + table_bbox[0],
                        text['y2'] + table_bbox[1]
                    ]
                    
                    
                    if col_idx+1 != len(row_texts):
                        if abs(row_texts[col_idx+1]['x']-text['x']) < 3 and abs(row_texts[col_idx+1]['y']-text['y']) > 3 and row_texts[col_idx+1]['y'] < text['y']:
                            text_in_cell=text_in_cell+" "+row_texts[col_idx+1]['text']+" "+text['text']
                            lswap=True
                        elif row_texts[col_idx+1]['x'] < text['x']:
                            text_in_cell=text_in_cell+" "+row_texts[col_idx+1]['text']+" "+text['text']
                            lswap=True
                        else:
                            if lswap:
                                lswap=False
                            else:
                                text_in_cell=text_in_cell+" "+text['text']    
                                

            cell_info_start = {
                        'row': row_idx,
                        'col': 0,
                        'text': text_in_cell,
                        'bbox': start_cell_bbox,
                        'confidence': 0.8  # Default confidence for text-based grid
                    }

            table_structure["cells"].append(cell_info_start)
            row_data.append(text_in_cell)
            table_structure["cells"].append(cell_info)
            row_data.append(end_text)
            
            table_structure["table_data"].append(row_data)
        
        return table_structure
    
    @staticmethod
    def _detect_cells_by_lines(gray: np.ndarray) -> List[Dict]:
        """Detect cells using line detection method"""
        # Apply thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Detect horizontal and vertical lines with different kernel sizes
        h_kernel_size = max(gray.shape[1] // 30, 15)  # Adaptive kernel size
        v_kernel_size = max(gray.shape[0] // 30, 15)
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
        
        # Extract lines
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find line intersections to create grid
        horizontal_coords = TableRecognition._find_line_coordinates(horizontal_lines, axis='horizontal')
        vertical_coords = TableRecognition._find_line_coordinates(vertical_lines, axis='vertical')
        
        # Create cells from grid intersections
        cells = []
        if len(horizontal_coords) >= 2 and len(vertical_coords) >= 2:
            for i in range(len(horizontal_coords) - 1):
                for j in range(len(vertical_coords) - 1):
                    x1, x2 = vertical_coords[j], vertical_coords[j + 1]
                    y1, y2 = horizontal_coords[i], horizontal_coords[i + 1]
                    
                    # Add some padding to avoid line pixels
                    padding = 2
                    x1, y1 = x1 + padding, y1 + padding
                    x2, y2 = x2 - padding, y2 - padding
                    
                    if x2 > x1 and y2 > y1:  # Valid cell
                        cells.append({
                            'x': x1, 'y': y1, 
                            'width': x2 - x1, 'height': y2 - y1,
                            'bbox': [x1, y1, x2, y2],
                            'area': (x2 - x1) * (y2 - y1)
                        })
        
        return cells
    
    @staticmethod
    def _find_line_coordinates(line_image: np.ndarray, axis: str) -> List[int]:
        """Find coordinates of detected lines"""
        coords = []
        
        if axis == 'horizontal':
            # Sum along horizontal axis to find horizontal lines
            projection = np.sum(line_image, axis=1)
            threshold = np.max(projection) * 0.3  # 30% of max
            
            for i, val in enumerate(projection):
                if val > threshold:
                    coords.append(i)
        else:  # vertical
            # Sum along vertical axis to find vertical lines
            projection = np.sum(line_image, axis=0)
            threshold = np.max(projection) * 0.3
            
            for i, val in enumerate(projection):
                if val > threshold:
                    coords.append(i)
        
        # Remove duplicate nearby coordinates
        if coords:
            filtered_coords = [coords[0]]
            for coord in coords[1:]:
                if coord - filtered_coords[-1] > 10:  # Minimum distance between lines
                    filtered_coords.append(coord)
            coords = filtered_coords
        
        return coords
    
    # @staticmethod
    # def _detect_cells_by_grid(gray: np.ndarray, image_shape: tuple) -> List[Dict]:
    #     """Fallback: Create grid based on text distribution"""
    #     height, width = image_shape[:2]
        
    #     # Create a simple 3x3 grid as fallback
    #     rows, cols = 3, 3
    #     cell_height = height // rows
    #     cell_width = width // cols
        
    #     cells = []
    #     for row in range(rows):
    #         for col in range(cols):
    #             x = col * cell_width
    #             y = row * cell_height
    #             w = cell_width if col < cols - 1 else width - x
    #             h = cell_height if row < rows - 1 else height - y
                
    #             cells.append({
    #                 'x': x, 'y': y, 'width': w, 'height': h,
    #                 'bbox': [x, y, x + w, y + h],
    #                 'area': w * h
    #             })
        
    #     return cells
    
    @staticmethod
    def _map_cells_to_full_page(cells: List[Dict], layout_box: Dict, 
                               page_data: Dict) -> List[Dict]:
        """Map detected cell coordinates to full page coordinates"""
        table_bbox = layout_box['bbox']
        table_x_offset = table_bbox[0]
        table_y_offset = table_bbox[1]
        
        mapped_cells = []
        for cell in cells:
            mapped_cell = cell.copy()
            # Add table offset to get full page coordinates
            mapped_cell['full_page_bbox'] = [
                cell['bbox'][0] + table_x_offset,
                cell['bbox'][1] + table_y_offset,
                cell['bbox'][2] + table_x_offset,
                cell['bbox'][3] + table_y_offset
            ]
            mapped_cell['full_page_x'] = cell['x'] + table_x_offset
            mapped_cell['full_page_y'] = cell['y'] + table_y_offset
            mapped_cells.append(mapped_cell)
        
        return mapped_cells
    
    @staticmethod
    def _assign_text_to_cells(cells: List[Dict], text_lines: List[dict]) -> List[Dict]:
        """Assign text content to cells and calculate confidence scores"""
        for cell in cells:
            cell['texts'] = []
            cell['confidence_scores'] = []
            
            cell_bbox = cell['full_page_bbox']
            
            for text_data in text_lines:
                text_bbox = text_data['bbox']
                
                # Calculate intersection over union (IoU) and overlap percentage
                overlap_score = TableRecognition._calculate_text_cell_overlap(
                    text_bbox, cell_bbox
                )
                
                # If significant overlap, assign text to cell
                if overlap_score > 0.1:  # 10% overlap threshold
                    cell['texts'].append(text_data['text'])
                    cell['confidence_scores'].append({
                        'text_confidence': text_data['confidence'],
                        'overlap_score': overlap_score,
                        'combined_score': text_data['confidence'] * overlap_score
                    })
        
        return cells
    
    @staticmethod
    def _calculate_text_cell_overlap(text_bbox: List[float], 
                                   cell_bbox: List[float]) -> float:
        """Calculate overlap between text bbox and cell bbox"""
        # Calculate intersection
        x1 = max(text_bbox[0], cell_bbox[0])
        y1 = max(text_bbox[1], cell_bbox[1])
        x2 = min(text_bbox[2], cell_bbox[2])
        y2 = min(text_bbox[3], cell_bbox[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        text_area = (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1])
        
        # Return intersection over text area (what percentage of text is in cell)
        return intersection_area / text_area if text_area > 0 else 0.0
    
    @staticmethod
    def _structure_table_data(cells: List[Dict]) -> Dict:
        """Structure the detected cells into a table format"""
        if not cells:
            return {"error": "No cells detected", "rows": 0, "cols": 0, "cells": []}
        
        # Group cells by rows and columns based on Y and X coordinates
        # Sort by Y coordinate to get rows
        cells_sorted_y = sorted(cells, key=lambda c: c['full_page_y'])
        
        # Group into rows (cells with similar Y coordinates)
        rows = []
        current_row = []
        row_threshold = 20  # pixels threshold for same row
        
        for cell in cells_sorted_y:
            if not current_row:
                current_row.append(cell)
            else:
                # Check if cell is in same row as previous cells
                avg_y = sum(c['full_page_y'] for c in current_row) / len(current_row)
                if abs(cell['full_page_y'] - avg_y) <= row_threshold:
                    current_row.append(cell)
                else:
                    # Sort current row by X coordinate
                    current_row.sort(key=lambda c: c['full_page_x'])
                    rows.append(current_row)
                    current_row = [cell]
        
        # Don't forget the last row
        if current_row:
            current_row.sort(key=lambda c: c['full_page_x'])
            rows.append(current_row)
        
        # Create structured table
        table_structure = {
            "rows": len(rows),
            "cols": max(len(row) for row in rows) if rows else 0,
            "cells": [],
            "table_data": []
        }
        
        # Add row and column indices
        for row_idx, row in enumerate(rows):
            row_data = []
            for col_idx, cell in enumerate(row):
                cell['row'] = row_idx
                cell['col'] = col_idx
                
                # Combine all texts in the cell
                combined_text = ' '.join(cell.get('texts', []))
                
                cell_info = {
                    'row': row_idx,
                    'col': col_idx,
                    'text': combined_text,
                    'bbox': cell['full_page_bbox'],
                    'confidence': max([score['combined_score'] 
                                     for score in cell.get('confidence_scores', [])], 
                                    default=0.0)
                }
                
                table_structure["cells"].append(cell_info)
                row_data.append(combined_text)
            
            table_structure["table_data"].append(row_data)
        
        return table_structure


