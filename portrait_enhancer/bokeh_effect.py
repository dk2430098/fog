"""
Bokeh Effect Module
Creates professional background blur (bokeh) effect.
"""

import cv2
import numpy as np


def create_circular_kernel(size):
    """
    Create a circular kernel for realistic bokeh effect.
    
    Args:
        size: Kernel size (should be odd)
    
    Returns:
        Circular kernel
    """
    if size % 2 == 0:
        size += 1
    
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    radius = size // 2
    
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist <= radius:
                kernel[i, j] = 1.0
    
    # Normalize
    kernel /= kernel.sum()
    
    return kernel


def apply_variable_blur(image, mask, blur_strength=21):
    """
    Apply variable blur based on mask (stronger blur in background).
    
    Args:
        image: Input image
        mask: Foreground mask (1 = foreground, 0 = background)
        blur_strength: Maximum blur kernel size for background
    
    Returns:
        Image with background blur
    """
    # Ensure blur_strength is odd
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    # Create strongly blurred version for background
    blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    
    # Expand mask to 3 channels
    if len(mask.shape) == 2:
        mask_3ch = cv2.merge([mask, mask, mask])
    else:
        mask_3ch = mask
    
    # Blend original (foreground) with blurred (background)
    result = (image * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
    
    return result


def apply_bokeh_effect(image, mask, intensity='medium'):
    """
    Apply artistic bokeh effect to background.
    
    Args:
        image: Input image
        mask: Foreground mask (float, 0-1)
        intensity: 'light', 'medium', or 'strong'
    
    Returns:
        Image with bokeh effect
    """
    # Determine blur strength based on intensity
    intensity_map = {
        'light': 15,
        'medium': 25,
        'strong': 35
    }
    blur_strength = intensity_map.get(intensity, 25)
    
    # Apply variable blur
    result = apply_variable_blur(image, mask, blur_strength)
    
    # Add subtle brightness boost to background for professional look
    background_boost = 1.1
    background_mask = 1.0 - mask
    
    # Ensure background mask is 3 channels
    if len(background_mask.shape) == 2:
        background_mask_3ch = cv2.merge([background_mask, background_mask, background_mask])
    else:
        background_mask_3ch = background_mask
    
    # Ensure foreground mask is 3 channels for blending
    if len(mask.shape) == 2:
        mask_3ch = cv2.merge([mask, mask, mask])
    elif mask.shape[2] == 1:
        mask_3ch = cv2.merge([mask[:,:,0], mask[:,:,0], mask[:,:,0]])
    else:
        mask_3ch = mask
    
    boosted = (result * background_boost).clip(0, 255).astype(np.uint8)
    result = (result.astype(np.float32) * mask_3ch + boosted.astype(np.float32) * background_mask_3ch).astype(np.uint8)
    
    return result


def create_depth_blur(image, mask, face_bbox=None):
    """
    Create depth-based blur effect (stronger blur farther from subject).
    
    Args:
        image: Input image
        mask: Foreground mask
        face_bbox: Optional face bounding box for depth center
    
    Returns:
        Image with depth-based blur
    """
    h, w = image.shape[:2]
    
    # Create depth map (distance from center/face)
    if face_bbox is not None:
        x, y, fw, fh = face_bbox
        center_x = x + fw // 2
        center_y = y + fh // 2
    else:
        center_x = w // 2
        center_y = h // 2
    
    # Create coordinate grids
    y_grid, x_grid = np.ogrid[:h, :w]
    
    # Calculate distance from center
    distance = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    
    # Normalize distance to 0-1
    depth_map = distance / distance.max()
    
    # Apply stronger blur to areas farther from center
    # Create multiple blur levels
    blur_levels = [
        cv2.GaussianBlur(image, (15, 15), 0),
        cv2.GaussianBlur(image, (25, 25), 0),
        cv2.GaussianBlur(image, (35, 35), 0)
    ]
    
    # Blend based on depth
    result = image.copy().astype(np.float32)
    
    # Use depth map to blend blur levels
    depth_blur = (blur_levels[0] * 0.3 + 
                  blur_levels[1] * 0.4 + 
                  blur_levels[2] * 0.3).astype(np.float32)
    
    # Expand mask to 3 channels
    if len(mask.shape) == 2:
        mask_3ch = cv2.merge([mask, mask, mask])
    else:
        mask_3ch = mask
    
    # Blend with original foreground
    result = (image * mask_3ch + depth_blur * (1 - mask_3ch)).astype(np.uint8)
    
    return result


def test():
    """Test function for module validation."""
    # Create test image and mask
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_mask = np.ones((480, 640), dtype=np.float32)
    test_mask[100:380, 200:440] = 1.0  # Foreground region
    test_mask = cv2.GaussianBlur(test_mask, (21, 21), 0)
    
    # Test bokeh effect
    result = apply_bokeh_effect(test_image, test_mask, intensity='medium')
    print(f"Bokeh effect complete. Output shape: {result.shape}")
    
    # Test depth blur
    result2 = create_depth_blur(test_image, test_mask)
    print(f"Depth blur complete. Output shape: {result2.shape}")
    
    return True
