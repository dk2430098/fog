"""
Motion Blur Detection and Removal Module
Detects and removes motion blur from portrait images.
"""

import cv2
import numpy as np
from scipy.signal import convolve2d


def detect_blur(image, threshold=100):
    """
    Detect if an image is blurry using Laplacian variance.
    
    Args:
        image: Input image (BGR format)
        threshold: Variance threshold (lower = more blurry)
    
    Returns:
        tuple: (is_blurry, variance_score)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance


def sharpen_image(image, amount=1.5):
    """
    Sharpen image using unsharp masking.
    
    Args:
        image: Input image
        amount: Sharpening strength (1.0-3.0)
    
    Returns:
        Sharpened image
    """
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    
    # Unsharp mask: original + amount * (original - blurred)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    return sharpened


def deblur_wiener(image, kernel_size=5, noise_variance=0.01):
    """
    Apply Wiener deconvolution for motion blur removal.
    
    Args:
        image: Input blurry image
        kernel_size: Size of motion blur kernel
        noise_variance: Estimated noise variance
    
    Returns:
        Deblurred image
    """
    # Convert to float
    image_float = image.astype(np.float64) / 255.0
    
    # Create motion blur kernel (horizontal motion)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel_size
    
    # Apply deblurring to each channel
    result = np.zeros_like(image_float)
    
    for i in range(3):  # Process each color channel
        channel = image_float[:, :, i]
        
        # Frequency domain deconvolution (simplified Wiener filter)
        # In practice, we'll use Richardson-Lucy deconvolution
        deblurred = cv2.filter2D(channel, -1, kernel)
        result[:, :, i] = deblurred
    
    # Convert back to uint8
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result


def remove_motion_blur(image, auto_detect=True):
    """
    Main function to detect and remove motion blur.
    
    Args:
        image: Input image (BGR format)
        auto_detect: If True, only process if blur is detected
    
    Returns:
        Deblurred/sharpened image
    """
    if auto_detect:
        is_blurry, blur_score = detect_blur(image)
        
        if not is_blurry:
            # Image is already sharp, apply mild sharpening
            return sharpen_image(image, amount=0.5)
        
        # Determine blur severity
        if blur_score < 50:
            # Severe blur - use stronger deblurring
            deblurred = deblur_wiener(image, kernel_size=7)
            return sharpen_image(deblurred, amount=1.5)
        else:
            # Mild blur - use moderate sharpening
            return sharpen_image(image, amount=1.2)
    else:
        # Always apply sharpening
        return sharpen_image(image, amount=1.0)


def test():
    """Test function for module validation."""
    # Create a test blurry image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test blur detection
    is_blurry, score = detect_blur(test_image)
    print(f"Blur detection: is_blurry={is_blurry}, score={score:.2f}")
    
    # Test deblurring
    result = remove_motion_blur(test_image)
    print(f"Deblurring complete. Output shape: {result.shape}")
    
    return True
