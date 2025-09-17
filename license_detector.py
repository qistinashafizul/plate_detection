import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import re
import os
from typing import List, Tuple, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LicensePlateRecognizer:
    """
    Modern License Plate Recognizer using OpenCV + EasyOCR
    Much simpler and more accurate than the custom ML approach
    """
    
    def __init__(self, languages=['en'], gpu=False):
        """
        Initialize the recognizer
        
        Args:
            languages: List of languages for OCR (e.g., ['en'] for English)
            gpu: Whether to use GPU acceleration
        """
        print("🚀 Initializing Modern License Plate Recognizer...")
        
        try:
            # Initialize EasyOCR reader
            self.reader = easyocr.Reader(languages, gpu=gpu)
            print("✅ EasyOCR initialized successfully")
            
            # Initialize cascade classifier for plate detection (optional)
            self.plate_cascade = None
            self._try_load_cascade()
            
            # Configuration
            self.min_confidence = 0.1
            self.min_plate_area = 500
            self.max_plate_area = 100000
            
        except Exception as e:
            print(f"❌ Error initializing EasyOCR: {e}")
            raise
    
    def _try_load_cascade(self):
        """Try to load Haar cascade for plate detection (optional enhancement)"""
        try:
            # You can download this from OpenCV's GitHub repository
            cascade_path = 'haarcascade_russian_plate_number.xml'
            if os.path.exists(cascade_path):
                self.plate_cascade = cv2.CascadeClassifier(cascade_path)
                print("✅ Haar cascade loaded for enhanced plate detection")
        except Exception:
            print("ℹ️ Haar cascade not available, using contour-based detection")
    
    def detect_plates_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect license plate regions using OpenCV contour detection
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of bounding boxes (x, y, w, h) for potential plates
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Find edges using Canny
        edges = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        
        plate_candidates = []
        
        for contour in contours:
            # Approximate the contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h
            
            # Filter based on typical license plate characteristics
            if (self.min_plate_area < area < self.max_plate_area and
                2.0 < aspect_ratio < 6.0 and  # Typical plate aspect ratio
                h > 15 and w > 50):           # Minimum size
                
                plate_candidates.append((x, y, w, h))
        
        return plate_candidates
    
    def detect_plates_cascade(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect plates using Haar cascade (if available)
        """
        if self.plate_cascade is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 15)
        )
        
        return [(x, y, w, h) for (x, y, w, h) in plates]
    
    def preprocess_plate_region(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Preprocess the plate region for better OCR results
        
        Args:
            plate_image: Cropped plate region
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Resize if too small (EasyOCR works better with larger images)
        height, width = denoised.shape
        if height < 50 or width < 150:
            scale_factor = max(50 / height, 150 / width, 2.0)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            denoised = cv2.resize(denoised, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return denoised
    
    def extract_text_easyocr(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text using EasyOCR
        
        Args:
            image: Image to extract text from
            
        Returns:
            List of detection results
        """
        try:
            # EasyOCR expects RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Run OCR
            results = self.reader.readtext(image_rgb)
            
            # Parse results
            detections = []
            for (bbox, text, confidence) in results:
                if confidence > self.min_confidence:
                    detections.append({
                        'text': text,
                        'confidence': confidence,
                        'bbox': bbox
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []
    
    def clean_license_plate_text(self, text: str) -> str:
        """
        Clean and validate license plate text
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned license plate text
        """
        # Remove spaces and convert to uppercase
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Apply common OCR corrections
        corrections = {
            'O': '0',  # Letter O to number 0
            'I': '1',  # Letter I to number 1  
            'S': '5',  # Letter S to number 5 (sometimes)
            'Z': '2',  # Letter Z to number 2 (sometimes)
            'B': '8',  # Letter B to number 8 (sometimes)
        }
        
        # Apply corrections selectively based on position
        # (You can customize this based on your country's license plate format)
        result = ""
        for i, char in enumerate(cleaned):
            if i >= len(cleaned) - 4:  # Last 4 positions usually numbers
                result += corrections.get(char, char) if char in corrections and corrections[char].isdigit() else char
            else:  # First positions usually letters
                result += char
        
        return result
    
    def validate_license_plate(self, text: str) -> bool:
        """
        Validate if text looks like a license plate
        
        Args:
            text: Text to validate
            
        Returns:
            True if text looks like a valid license plate
        """
        # Remove spaces and special characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Check length (most plates are 4-10 characters)
        if not (4 <= len(cleaned) <= 10):
            return False
        
        # Check if it contains both letters and numbers (most plates do)
        has_letters = bool(re.search(r'[A-Z]', cleaned))
        has_numbers = bool(re.search(r'[0-9]', cleaned))
        
        # At least one letter OR at least one number
        if not (has_letters or has_numbers):
            return False
        
        # Reject if it's all the same character
        if len(set(cleaned)) <= 1:
            return False
        
        return True
    
    def recognize_from_image_path(self, image_path: str) -> Dict:
        """
        Recognize license plates from image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with results
        """
        if not os.path.exists(image_path):
            return {'error': f'Image not found: {image_path}'}
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': f'Could not read image: {image_path}'}
            
            return self.recognize_from_image(image)
            
        except Exception as e:
            return {'error': f'Error processing image: {str(e)}'}
    
    def recognize_from_image(self, image: np.ndarray) -> Dict:
        """
        Main recognition function
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with recognition results
        """
        results = {
            'plates': [],
            'best_result': None,
            'processing_info': []
        }
        
        try:
            original_image = image.copy()
            
            # Method 1: Try direct OCR on the whole image first
            results['processing_info'].append("🔍 Trying direct OCR on full image...")
            direct_ocr_results = self.extract_text_easyocr(image)
            
            for detection in direct_ocr_results:
                cleaned_text = self.clean_license_plate_text(detection['text'])
                if self.validate_license_plate(cleaned_text) and len(cleaned_text) >= 4:
                    results['plates'].append({
                        'text': cleaned_text,
                        'confidence': detection['confidence'],
                        'method': 'direct_ocr',
                        'bbox': detection['bbox']
                    })
            
            # Method 2: Detect plate regions first, then OCR
            results['processing_info'].append("🎯 Detecting plate regions...")
            
            # Try OpenCV contour detection
            plate_regions = self.detect_plates_opencv(image)
            
            # Also try cascade detection if available
            if self.plate_cascade is not None:
                cascade_regions = self.detect_plates_cascade(image)
                plate_regions.extend(cascade_regions)
            
            results['processing_info'].append(f"📍 Found {len(plate_regions)} potential plate regions")
            
            # Process each detected region
            for i, (x, y, w, h) in enumerate(plate_regions):
                # Extract plate region
                plate_roi = original_image[y:y+h, x:x+w]
                
                # Preprocess for better OCR
                processed_plate = self.preprocess_plate_region(plate_roi)
                
                # Run OCR on the plate region
                plate_ocr_results = self.extract_text_easyocr(processed_plate)
                
                for detection in plate_ocr_results:
                    cleaned_text = self.clean_license_plate_text(detection['text'])
                    if self.validate_license_plate(cleaned_text) and len(cleaned_text) >= 4:
                        results['plates'].append({
                            'text': cleaned_text,
                            'confidence': detection['confidence'],
                            'method': f'region_ocr_{i}',
                            'bbox': [(x, y), (x+w, y), (x+w, y+h), (x, y+h)],  # Convert to EasyOCR format
                            'region': (x, y, w, h)
                        })
            
            # Find the best result
            if results['plates']:
                # Sort by confidence
                results['plates'].sort(key=lambda x: x['confidence'], reverse=True)
                results['best_result'] = results['plates'][0]
                results['processing_info'].append(f"✅ Best result: '{results['best_result']['text']}' (confidence: {results['best_result']['confidence']:.2f})")
            else:
                results['processing_info'].append("❌ No valid license plates detected")
            
            return results
            
        except Exception as e:
            results['error'] = f"Recognition failed: {str(e)}"
            results['processing_info'].append(f"❌ Error: {str(e)}")
            return results
    
    def visualize_results(self, image: np.ndarray, results: Dict, save_path: str = None) -> None:
        """
        Visualize detection results
        
        Args:
            image: Original image
            results: Results from recognition
            save_path: Optional path to save visualization
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display image
        if len(image.shape) == 3:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(image, cmap='gray')
        
        # Draw bounding boxes for all detections
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        
        for i, plate in enumerate(results.get('plates', [])):
            color = colors[i % len(colors)]
            
            if 'region' in plate:
                # Rectangle region
                x, y, w, h = plate['region']
                rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y-10, f"{plate['text']} ({plate['confidence']:.2f})", 
                       color=color, fontsize=12, fontweight='bold')
            elif 'bbox' in plate:
                # Polygon bbox from direct OCR
                bbox = np.array(plate['bbox'])
                if len(bbox) == 4:  # 4 corner points
                    # Draw polygon
                    polygon = plt.Polygon(bbox, fill=False, edgecolor=color, linewidth=2)
                    ax.add_patch(polygon)
                    
                    # Add text at top-left corner
                    x, y = bbox[0]
                    ax.text(x, y-10, f"{plate['text']} ({plate['confidence']:.2f})", 
                           color=color, fontsize=12, fontweight='bold')
        
        ax.set_title('License Plate Recognition Results')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"📊 Results saved to {save_path}")
        
        plt.show()
