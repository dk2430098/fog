"""
Face Enhancement Module
Improves face clarity while preserving natural skin texture.
Uses OpenCV Haar Cascade for face detection.
"""

import cv2
import numpy as np
import os


class FaceEnhancer:
    """Handles face detection and enhancement."""
    
    def __init__(self):
        """Initialize OpenCV Face Detection with Haar Cascade."""
        # Load Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect_faces(self, image):
        """
        Detect faces in the image.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of face bounding boxes [(x, y, w, h), ...]
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list of tuples
        face_list = []
        for (x, y, w, h) in faces:
            # Ensure coordinates are within image bounds
            h_img, w_img = image.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            
            face_list.append((x, y, w, h))
        
        return face_list
    
    def smooth_skin(self, image, face_bbox, strength=0.3):
        """
        Apply gentle skin smoothing using bilateral filter.
        
        Args:
            image: Input image
            face_bbox: Face bounding box (x, y, w, h)
            strength: Smoothing strength (0.0-1.0)
        
        Returns:
            Image with smoothed skin in face region
        """
        x, y, w, h = face_bbox
        
        # Extract face region with padding to avoid edge artifacts
        face_region = image[y:y+h, x:x+w].copy()
        
        # Apply gentle bilateral filter - reduced parameters to prevent black spots
        d = max(5, int(7 * strength))  # Smaller filter diameter
        sigma_color = 40 * strength  # Reduced from 75
        sigma_space = 40 * strength  # Reduced from 75
        
        smoothed = cv2.bilateralFilter(face_region, d, sigma_color, sigma_space)
        
        # Blend smoothed with original to prevent over-smoothing
        # Use weighted blend instead of frequency separation to avoid artifacts
        blend_ratio = 0.4 * strength  # Much gentler blend
        result = cv2.addWeighted(smoothed, blend_ratio, face_region, 1.0 - blend_ratio, 0)
        
        # Ensure no negative values or overflow
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Place back into image
        output = image.copy()
        output[y:y+h, x:x+w] = result
        
        return output
    
    def enhance_details(self, image, face_bbox, amount=1.15):
        """
        Gently enhance facial details using local contrast enhancement.
        
        Args:
            image: Input image
            face_bbox: Face bounding box
            amount: Enhancement amount (1.0-1.5) - reduced to prevent artifacts
        
        Returns:
            Image with enhanced facial details
        """
        x, y, w, h = face_bbox
        
        # Extract face region
        face_region = image[y:y+h, x:x+w].copy()
        
        # Convert to LAB color space
        lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply gentle CLAHE to L channel - reduced clip limit to prevent black spots
        clahe = cv2.createCLAHE(clipLimit=1.5 * amount, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Blend enhanced L with original to prevent over-enhancement
        l_blended = cv2.addWeighted(l_enhanced, 0.6, l, 0.4, 0)
        
        # Merge back
        enhanced_lab = cv2.merge([l_blended, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Place back into image
        output = image.copy()
        output[y:y+h, x:x+w] = enhanced
        
        return output
    
    def enhance_face(self, image, smooth_strength=0.25, detail_amount=1.1):
        """
        Complete face enhancement pipeline with gentle parameters.
        
        Args:
            image: Input image
            smooth_strength: Skin smoothing strength (reduced to prevent artifacts)
            detail_amount: Detail enhancement amount (reduced to prevent black spots)
        
        Returns:
            Enhanced image
        """
        # Detect faces
        faces = self.detect_faces(image)
        
        if not faces:
            # No faces detected, return original
            return image
        
        # Process each detected face
        result = image.copy()
        for face_bbox in faces:
            # Apply skin smoothing
            result = self.smooth_skin(result, face_bbox, smooth_strength)
            
            # Enhance details
            result = self.enhance_details(result, face_bbox, detail_amount)
        
        return result
    
    def close(self):
        """Release resources."""
        pass


def test():
    """Test function for module validation."""
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize enhancer
    enhancer = FaceEnhancer()
    
    # Test face detection
    faces = enhancer.detect_faces(test_image)
    print(f"Detected {len(faces)} faces")
    
    # Test enhancement
    result = enhancer.enhance_face(test_image)
    print(f"Enhancement complete. Output shape: {result.shape}")
    
    # Cleanup
    enhancer.close()
    
    return True
