"""
Sharpness and Contrast Enhancement Module
Improves overall image quality through sharpening and contrast adjustment.
"""

import cv2
import numpy as np


def unsharp_mask(image, amount=1.5, radius=2.0, threshold=0):
    """
    Apply unsharp masking for sharpness enhancement.
    
    Args:
        image: Input image
        amount: Sharpening strength (0.5-3.0)
        radius: Blur radius for mask creation
        threshold: Minimum brightness change to sharpen
    
    Returns:
        Sharpened image
    """
    # Create blurred version
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    
    # Calculate sharpening mask
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    # Apply threshold if specified
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        sharpened = np.where(low_contrast_mask, image, sharpened)
    
    return sharpened


def enhance_contrast(image, clip_limit=2.0, tile_size=8):
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image
        clip_limit: Contrast limiting threshold
        tile_size: Grid size for local enhancement
    
    Returns:
        Contrast-enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)
    
    # Merge back
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return result


def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    Adjust brightness and contrast.
    
    Args:
        image: Input image
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast adjustment (-100 to 100)
    
    Returns:
        Adjusted image
    """
    # Convert to float for calculations
    img_float = image.astype(np.float32)
    
    # Apply contrast adjustment
    contrast_factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
    img_float = contrast_factor * (img_float - 128) + 128
    
    # Apply brightness adjustment
    img_float = img_float + brightness
    
    # Clip values
    result = np.clip(img_float, 0, 255).astype(np.uint8)
    
    return result


def enhance_saturation(image, saturation_boost=1.2):
    """
    Enhance color saturation.
    
    Args:
        image: Input image
        saturation_boost: Saturation multiplier (1.0 = no change)
    
    Returns:
        Saturation-enhanced image
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    # Enhance saturation
    s = s * saturation_boost
    s = np.clip(s, 0, 255)
    
    # Merge back
    enhanced_hsv = cv2.merge([h, s, v]).astype(np.uint8)
    result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    
    return result


def apply_tone_curve(image, shadows=0, midtones=0, highlights=0):
    """
    Apply tone curve adjustments for professional color grading.
    
    Args:
        image: Input image
        shadows: Shadow adjustment (-50 to 50)
        midtones: Midtone adjustment (-50 to 50)
        highlights: Highlight adjustment (-50 to 50)
    
    Returns:
        Tone-adjusted image
    """
    # Create lookup table for tone curve
    lut = np.arange(256, dtype=np.float32)
    
    # Apply shadow adjustment (affects dark tones)
    shadow_curve = np.power(lut / 255.0, 1.0 - shadows/100.0) * 255
    
    # Apply midtone adjustment
    midtone_curve = lut + midtones * (1.0 - np.abs(lut - 128) / 128)
    
    # Apply highlight adjustment (affects bright tones)
    highlight_curve = 255 - np.power((255 - lut) / 255.0, 1.0 - highlights/100.0) * 255
    
    # Combine curves
    final_lut = (shadow_curve * 0.33 + midtone_curve * 0.34 + highlight_curve * 0.33)
    final_lut = np.clip(final_lut, 0, 255).astype(np.uint8)
    
    # Apply lookup table to each channel
    result = cv2.LUT(image, final_lut)
    
    return result


def enhance_image_quality(image, sharpen_amount=1.5, contrast_boost=2.0, 
                         saturation_boost=1.15, brightness=5):
    """
    Complete image quality enhancement pipeline.
    
    Args:
        image: Input image
        sharpen_amount: Sharpening strength
        contrast_boost: CLAHE clip limit
        saturation_boost: Saturation multiplier
        brightness: Brightness adjustment
    
    Returns:
        Enhanced image
    """
    # Apply sharpening
    result = unsharp_mask(image, amount=sharpen_amount)
    
    # Enhance contrast
    result = enhance_contrast(result, clip_limit=contrast_boost)
    
    # Adjust brightness slightly
    if brightness != 0:
        result = adjust_brightness_contrast(result, brightness=brightness, contrast=0)
    
    # Enhance saturation
    result = enhance_saturation(result, saturation_boost=saturation_boost)
    
    # Apply subtle tone curve for professional look
    result = apply_tone_curve(result, shadows=5, midtones=0, highlights=5)
    
    return result


def test():
    """Test function for module validation."""
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test sharpening
    sharpened = unsharp_mask(test_image)
    print(f"Sharpening complete. Output shape: {sharpened.shape}")
    
    # Test contrast enhancement
    contrasted = enhance_contrast(test_image)
    print(f"Contrast enhancement complete. Output shape: {contrasted.shape}")
    
    # Test full enhancement
    enhanced = enhance_image_quality(test_image)
    print(f"Full enhancement complete. Output shape: {enhanced.shape}")
    
    return True
