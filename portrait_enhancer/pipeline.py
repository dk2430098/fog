"""
Main Processing Pipeline
Orchestrates all enhancement steps in optimal order.
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional

from . import blur_removal
from . import segmentation
from . import face_enhancement
from . import bokeh_effect
from . import sharpness


class PortraitEnhancer:
    """Main class for portrait enhancement pipeline."""
    
    def __init__(self, verbose=True):
        """
        Initialize the portrait enhancer.
        
        Args:
            verbose: If True, print processing steps and timing
        """
        self.verbose = verbose
        self.segmenter = segmentation.PersonSegmenter()
        self.face_enhancer = face_enhancement.FaceEnhancer()
        self.processing_times = {}
    
    def process_image(self, image, 
                     remove_blur=True,
                     add_bokeh=True,
                     enhance_face=True,
                     boost_sharpness=True,
                     bokeh_intensity='medium'):
        """
        Process a portrait image with all enhancements.
        
        Args:
            image: Input image (BGR format)
            remove_blur: Apply motion blur removal
            add_bokeh: Add background blur effect
            enhance_face: Apply face enhancement
            boost_sharpness: Apply sharpness and contrast boost
            bokeh_intensity: 'light', 'medium', or 'strong'
        
        Returns:
            Enhanced image
        """
        if self.verbose:
            print("=" * 60)
            print("Starting Portrait Enhancement Pipeline")
            print("=" * 60)
        
        start_time = time.time()
        result = image.copy()
        
        # Step 1: Remove motion blur (if present)
        if remove_blur:
            step_start = time.time()
            if self.verbose:
                print("\n[1/5] Removing motion blur and initial sharpening...")
            
            result = blur_removal.remove_motion_blur(result)
            self.processing_times['blur_removal'] = time.time() - step_start
            
            if self.verbose:
                print(f"      ✓ Completed in {self.processing_times['blur_removal']:.3f}s")
        
        # Step 2: Segment foreground/background
        step_start = time.time()
        if self.verbose:
            print("\n[2/5] Segmenting person from background...")
        
        fg_mask, bg_mask, feathered_mask = self.segmenter.get_segmentation_masks(result)
        self.processing_times['segmentation'] = time.time() - step_start
        
        if self.verbose:
            print(f"      ✓ Completed in {self.processing_times['segmentation']:.3f}s")
        
        # Step 3: Apply bokeh effect to background
        if add_bokeh:
            step_start = time.time()
            if self.verbose:
                print(f"\n[3/5] Adding {bokeh_intensity} bokeh effect to background...")
            
            result = bokeh_effect.apply_bokeh_effect(result, feathered_mask, 
                                                     intensity=bokeh_intensity)
            self.processing_times['bokeh'] = time.time() - step_start
            
            if self.verbose:
                print(f"      ✓ Completed in {self.processing_times['bokeh']:.3f}s")
        
        # Step 4: Enhance face clarity and skin texture
        if enhance_face:
            step_start = time.time()
            if self.verbose:
                print("\n[4/5] Enhancing face clarity and skin texture...")
            
            result = self.face_enhancer.enhance_face(result, 
                                                     smooth_strength=0.25,
                                                     detail_amount=1.1)
            self.processing_times['face_enhancement'] = time.time() - step_start
            
            if self.verbose:
                print(f"      ✓ Completed in {self.processing_times['face_enhancement']:.3f}s")
        
        # Step 5: Final sharpness and contrast boost
        if boost_sharpness:
            step_start = time.time()
            if self.verbose:
                print("\n[5/5] Applying final sharpness and contrast enhancement...")
            
            result = sharpness.enhance_image_quality(result,
                                                     sharpen_amount=1.3,
                                                     contrast_boost=1.8,
                                                     saturation_boost=1.15,
                                                     brightness=5)
            self.processing_times['sharpness'] = time.time() - step_start
            
            if self.verbose:
                print(f"      ✓ Completed in {self.processing_times['sharpness']:.3f}s")
        
        # Calculate total time
        total_time = time.time() - start_time
        self.processing_times['total'] = total_time
        
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"✓ Enhancement Complete! Total time: {total_time:.3f}s")
            print("=" * 60)
        
        return result
    
    def get_timing_report(self):
        """
        Get detailed timing report.
        
        Returns:
            Dictionary with processing times for each step
        """
        return self.processing_times.copy()
    
    def close(self):
        """Release resources."""
        self.segmenter.close()
        self.face_enhancer.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def enhance_portrait(input_path, output_path, **kwargs):
    """
    Convenience function to enhance a portrait image.
    
    Args:
        input_path: Path to input image
        output_path: Path to save enhanced image
        **kwargs: Additional arguments for PortraitEnhancer.process_image()
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not read image from {input_path}")
            return False
        
        # Process with enhancer
        with PortraitEnhancer(verbose=kwargs.get('verbose', True)) as enhancer:
            result = enhancer.process_image(image, **kwargs)
        
        # Save result
        cv2.imwrite(output_path, result)
        print(f"\n✓ Saved enhanced image to: {output_path}")
        
        return True
    
    except Exception as e:
        print(f"Error during enhancement: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test():
    """Test function for module validation."""
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize enhancer
    with PortraitEnhancer(verbose=False) as enhancer:
        # Test full pipeline
        result = enhancer.process_image(test_image)
        
        # Get timing report
        times = enhancer.get_timing_report()
        
        print("Pipeline test complete:")
        print(f"  Output shape: {result.shape}")
        print(f"  Total time: {times.get('total', 0):.3f}s")
    
    return True
