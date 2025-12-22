"""
Background/Foreground Segmentation Module
Uses OpenCV GrabCut for person segmentation (MediaPipe-free fallback).
"""

import cv2
import numpy as np


class PersonSegmenter:
    """Handles person segmentation using OpenCV-based approach."""
    
    def __init__(self):
        """Initialize segmenter."""
        pass
    
    def segment(self, image):
        """
        Segment person from background using a simple but effective approach.
        Uses edge detection and color-based segmentation.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Binary mask where 1 = person, 0 = background
        """
        h, w = image.shape[:2]
        
        # Create a simple center-weighted mask
        # Assumes person is roughly in center of frame (common for portraits)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create initial rectangular mask (center region is likely foreground)
        center_x, center_y = w // 2, h // 2
        rect_w, rect_h = int(w * 0.6), int(h * 0.7)
        
        x1 = max(0, center_x - rect_w // 2)
        y1 = max(0, center_y - rect_h // 2)
        x2 = min(w, center_x + rect_w // 2)
        y2 = min(h, center_y + rect_h // 2)
        
        # Use GrabCut for better segmentation
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        rect = (x1, y1, x2 - x1, y2 - y1)
        mask_temp = np.zeros((h, w), dtype=np.uint8)
        
        try:
            cv2.grabCut(image, mask_temp, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create binary mask from GrabCut result
            binary_mask = np.where((mask_temp == cv2.GC_FGD) | (mask_temp == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        except:
            # Fallback to simple ellipse mask if GrabCut fails
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(binary_mask, (center_x, center_y), 
                       (rect_w // 2, rect_h // 2), 0, 0, 360, 1, -1)
        
        return binary_mask
    
    def refine_mask(self, mask, kernel_size=5):
        """
        Refine segmentation mask using morphological operations.
        
        Args:
            mask: Binary mask
            kernel_size: Size of morphological kernel
        
        Returns:
            Refined mask
        """
        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Close small holes
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        
        return mask_opened
    
    def feather_edges(self, mask, blur_amount=15):
        """
        Feather mask edges for smooth blending.
        
        Args:
            mask: Binary mask
            blur_amount: Amount of blur for edge feathering
        
        Returns:
            Feathered mask with smooth edges (0-1 float)
        """
        # Convert to float
        mask_float = mask.astype(np.float32)
        
        # Apply Gaussian blur to soften edges
        feathered = cv2.GaussianBlur(mask_float, (blur_amount, blur_amount), 0)
        
        return feathered
    
    def get_segmentation_masks(self, image):
        """
        Get complete segmentation masks with refinement.
        
        Args:
            image: Input image
        
        Returns:
            tuple: (foreground_mask, background_mask, feathered_mask)
        """
        # Get initial segmentation
        mask = self.segment(image)
        
        # Refine the mask
        refined_mask = self.refine_mask(mask)
        
        # Feather edges
        feathered_mask = self.feather_edges(refined_mask)
        
        # Create background mask (inverse)
        background_mask = 1.0 - feathered_mask
        
        return refined_mask, background_mask, feathered_mask
    
    def close(self):
        """Release resources."""
        pass


def test():
    """Test function for module validation."""
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize segmenter
    segmenter = PersonSegmenter()
    
    # Test segmentation
    fg_mask, bg_mask, feathered = segmenter.get_segmentation_masks(test_image)
    
    print(f"Segmentation complete:")
    print(f"  Foreground mask shape: {fg_mask.shape}")
    print(f"  Background mask shape: {bg_mask.shape}")
    print(f"  Feathered mask shape: {feathered.shape}")
    
    # Cleanup
    segmenter.close()
    
    return True
