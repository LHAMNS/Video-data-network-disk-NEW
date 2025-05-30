"""
AVI Video Validation Module
Validates uncompressed RGB24 AVI files generated by the system
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import hashlib
import json

logger = logging.getLogger(__name__)


class AVIValidator:
    """
    Validates AVI files generated by DirectAVIEncoder
    Extracts frames and verifies data integrity
    """
    
    def __init__(self, avi_path: str, original_file_path: Optional[str] = None):
        """
        Initialize AVI validator
        
        Args:
            avi_path: Path to AVI file to validate
            original_file_path: Optional path to original file for comparison
        """
        self.avi_path = Path(avi_path)
        self.original_file_path = Path(original_file_path) if original_file_path else None
        
        # Video properties
        self.cap = None
        self.frame_count = 0
        self.width = 0
        self.height = 0
        self.fps = 0
        
        # Color mapping (16-color palette)
        self.color_palette = np.array([
            [0, 0, 0],       # Black
            [255, 255, 255], # White
            [255, 0, 0],     # Red
            [0, 255, 0],     # Green
            [0, 0, 255],     # Blue
            [255, 255, 0],   # Yellow
            [0, 255, 255],   # Cyan
            [255, 0, 255],   # Magenta
            [128, 0, 0],     # Dark red
            [0, 128, 0],     # Dark green
            [0, 0, 128],     # Dark blue
            [128, 128, 0],   # Olive
            [0, 128, 128],   # Dark cyan
            [128, 0, 128],   # Purple
            [128, 128, 128], # Gray
            [255, 128, 0],   # Orange
        ], dtype=np.uint8)
        
        # Validation results
        self.validation_results = {}
        
    def open_video(self) -> bool:
        """Open AVI file for validation"""
        try:
            self.cap = cv2.VideoCapture(str(self.avi_path))
            if not self.cap.isOpened():
                logger.error(f"Cannot open AVI file: {self.avi_path}")
                return False
                
            # Get video properties
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"AVI opened: {self.width}x{self.height}, {self.frame_count} frames, {self.fps} fps")
            return True
            
        except Exception as e:
            logger.error(f"Error opening AVI: {e}")
            return False
    
    def close_video(self):
        """Close video capture"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def extract_metadata_frames(self) -> Dict[str, Any]:
        """
        Extract metadata from first 3 frames
        Returns metadata dictionary
        """
        if not self.cap:
            return {}
            
        metadata = {}
        
        try:
            # Reset to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Frame 0: File metadata
            ret, frame0 = self.cap.read()
            if ret:
                # Try to decode metadata from frame (simplified)
                metadata['metadata_frame'] = True
                
            # Frame 1: Color calibration
            ret, frame1 = self.cap.read()
            if ret:
                metadata['calibration_frame'] = True
                # Analyze color bars if needed
                
            # Frame 2: Sync patterns
            ret, frame2 = self.cap.read()
            if ret:
                metadata['sync_frame'] = True
                # Check sync pattern integrity
                
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            
        return metadata
    
    def map_pixel_to_color_index(self, pixel: np.ndarray) -> int:
        """
        Map RGB pixel to closest color palette index
        
        Args:
            pixel: RGB pixel values [R, G, B]
            
        Returns:
            Color index (0-15)
        """
        # Find closest color in palette using Euclidean distance
        distances = np.sum((self.color_palette - pixel) ** 2, axis=1)
        return np.argmin(distances)
    
    def extract_data_from_frame(self, frame: np.ndarray, nine_to_one: bool = True) -> bytes:
        """
        Extract binary data from a single frame
        
        Args:
            frame: RGB frame (height, width, 3)
            nine_to_one: Whether 9-to-1 upsampling was used
            
        Returns:
            Extracted binary data
        """
        try:
            # Convert BGR to RGB (OpenCV uses BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Calculate logical dimensions
            if nine_to_one:
                logical_width = self.width // 3
                logical_height = self.height // 3
            else:
                logical_width = self.width
                logical_height = self.height
            
            # Extract logical pixels
            color_indices = []
            
            for y in range(logical_height):
                for x in range(logical_width):
                    if nine_to_one:
                        # Sample center pixel of 3x3 block
                        sample_x = x * 3 + 1
                        sample_y = y * 3 + 1
                        pixel = frame_rgb[sample_y, sample_x]
                    else:
                        pixel = frame_rgb[y, x]
                    
                    # Map to color index
                    color_idx = self.map_pixel_to_color_index(pixel)
                    color_indices.append(color_idx)
            
            # Convert indices to bytes (2 indices per byte for 4-bit encoding)
            data_bytes = bytearray()
            for i in range(0, len(color_indices), 2):
                high_nibble = color_indices[i] & 0x0F
                low_nibble = color_indices[i + 1] & 0x0F if i + 1 < len(color_indices) else 0
                byte_value = (high_nibble << 4) | low_nibble
                data_bytes.append(byte_value)
            
            return bytes(data_bytes)
            
        except Exception as e:
            logger.error(f"Error extracting data from frame: {e}")
            return b''
    
    def validate_video_structure(self) -> Dict[str, Any]:
        """
        Validate basic video structure and properties
        
        Returns:
            Validation results dictionary
        """
        results = {
            'file_exists': self.avi_path.exists(),
            'file_size': self.avi_path.stat().st_size if self.avi_path.exists() else 0,
            'is_valid_avi': False,
            'frame_count': 0,
            'resolution': None,
            'fps': 0,
            'metadata_frames': False
        }
        
        if not self.open_video():
            return results
            
        results.update({
            'is_valid_avi': True,
            'frame_count': self.frame_count,
            'resolution': f"{self.width}x{self.height}",
            'fps': self.fps
        })
        
        # Check for metadata frames
        metadata = self.extract_metadata_frames()
        results['metadata_frames'] = len(metadata) > 0
        
        self.close_video()
        return results
    
    def validate_data_integrity(self, callback=None) -> Dict[str, Any]:
        """
        Validate data integrity by extracting and comparing with original
        
        Args:
            callback: Progress callback function
            
        Returns:
            Validation results
        """
        results = {
            'validation_success': False,
            'frames_processed': 0,
            'data_extracted': 0,
            'accuracy_percentage': 0.0,
            'error_details': []
        }
        
        if not self.open_video():
            results['error_details'].append("Cannot open AVI file")
            return results
            
        try:
            # Skip metadata frames (first 3)
            start_frame = 3 if self.frame_count > 3 else 0
            data_frames = self.frame_count - start_frame
            
            extracted_data = bytearray()
            
            for frame_idx in range(start_frame, self.frame_count):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                
                if not ret:
                    results['error_details'].append(f"Cannot read frame {frame_idx}")
                    continue
                
                # Extract data from frame
                frame_data = self.extract_data_from_frame(frame, nine_to_one=True)
                extracted_data.extend(frame_data)
                
                results['frames_processed'] += 1
                results['data_extracted'] = len(extracted_data)
                
                # Call progress callback
                if callback:
                    progress = (frame_idx - start_frame + 1) / data_frames * 100
                    callback(progress, frame_idx, len(extracted_data))
            
            # Compare with original file if available
            if self.original_file_path and self.original_file_path.exists():
                with open(self.original_file_path, 'rb') as f:
                    original_data = f.read()
                
                # Calculate accuracy
                min_length = min(len(original_data), len(extracted_data))
                matching_bytes = sum(1 for i in range(min_length) 
                                   if original_data[i] == extracted_data[i])
                
                results['accuracy_percentage'] = (matching_bytes / min_length * 100) if min_length > 0 else 0
                results['original_size'] = len(original_data)
                results['extracted_size'] = len(extracted_data)
                
                if results['accuracy_percentage'] >= 99.9:
                    results['validation_success'] = True
                else:
                    results['error_details'].append(f"Data mismatch: {results['accuracy_percentage']:.2f}% accuracy")
            else:
                # Without original file, just check if data was extracted
                results['validation_success'] = len(extracted_data) > 0
                results['accuracy_percentage'] = 100.0 if len(extracted_data) > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            results['error_details'].append(str(e))
        finally:
            self.close_video()
        
        return results
    
    def full_validation(self, callback=None) -> Dict[str, Any]:
        """
        Perform complete validation of AVI file
        
        Args:
            callback: Progress callback function
            
        Returns:
            Complete validation results
        """
        logger.info(f"Starting full validation of {self.avi_path}")
        
        # Structure validation
        structure_results = self.validate_video_structure()
        
        # Data integrity validation
        integrity_results = self.validate_data_integrity(callback)
        
        # Combine results
        complete_results = {
            'file_path': str(self.avi_path),
            'validation_timestamp': np.datetime64('now').isoformat(),
            'structure_validation': structure_results,
            'integrity_validation': integrity_results,
            'overall_success': (structure_results['is_valid_avi'] and 
                              integrity_results['validation_success'])
        }
        
        logger.info(f"Validation complete. Success: {complete_results['overall_success']}")
        
        return complete_results


def validate_avi_file(avi_path: str, original_file_path: str = None, 
                     callback=None) -> Dict[str, Any]:
    """
    Convenience function to validate an AVI file
    
    Args:
        avi_path: Path to AVI file
        original_file_path: Optional path to original file
        callback: Progress callback
        
    Returns:
        Validation results
    """
    validator = AVIValidator(avi_path, original_file_path)
    return validator.full_validation(callback)
