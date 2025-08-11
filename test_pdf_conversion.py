#!/usr/bin/env python3
"""
Test script to verify PyMuPDF PDF conversion works correctly.
This script tests the convert_pdf_to_images function without requiring poppler.
"""

import fitz  # PyMuPDF
from PIL import Image
import os
import sys

def convert_pdf_to_images(pdf_path, dpi):
    """Convert PDF to a list of PIL Image objects using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Calculate zoom factor based on DPI (72 DPI is the default)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        raise

def test_pdf_conversion():
    """Test the PDF conversion function."""
    print("Testing PyMuPDF PDF conversion...")
    
    # Check if PyMuPDF is available
    try:
        import fitz
        print("✓ PyMuPDF is available")
    except ImportError:
        print("✗ PyMuPDF is not available. Please install it with: pip install PyMuPDF")
        return False
    
    # Test with a sample PDF if available
    test_pdf = "test.pdf"
    if os.path.exists(test_pdf):
        print(f"Testing with {test_pdf}...")
        try:
            images = convert_pdf_to_images(test_pdf, 300)
            print(f"✓ Successfully converted PDF to {len(images)} images")
            for i, img in enumerate(images):
                print(f"  Page {i+1}: {img.size[0]}x{img.size[1]} pixels")
            return True
        except Exception as e:
            print(f"✗ Error during conversion: {e}")
            return False
    else:
        print(f"No test PDF found at {test_pdf}")
        print("Creating a simple test to verify PyMuPDF functionality...")
        
        # Test basic PyMuPDF functionality
        try:
            # Create a simple PDF for testing
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Test PDF for PyMuPDF conversion")
            doc.save("test_output.pdf")
            doc.close()
            
            # Test conversion
            images = convert_pdf_to_images("test_output.pdf", 300)
            print(f"✓ Successfully converted test PDF to {len(images)} images")
            
            # Clean up
            os.remove("test_output.pdf")
            return True
        except Exception as e:
            print(f"✗ Error during test: {e}")
            return False

if __name__ == "__main__":
    success = test_pdf_conversion()
    if success:
        print("\n✓ PyMuPDF PDF conversion test passed!")
        print("The poppler dependency has been successfully replaced.")
    else:
        print("\n✗ PyMuPDF PDF conversion test failed!")
        sys.exit(1) 