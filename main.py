#!/usr/bin/env python3
"""
Studio-Quality Portrait Converter - CLI Interface
Command-line tool for enhancing portrait images.
"""

import argparse
import cv2
import os
import sys
from pathlib import Path

from portrait_enhancer import PortraitEnhancer


def process_single_image(input_path, output_path, args):
    """Process a single portrait image."""
    print(f"\nProcessing: {input_path}")
    
    # Read image
    image = cv2.imread(input_path)
    if image is None:
        print(f"ERROR: Could not read image: {input_path}")
        return False
    
    # Process image
    with PortraitEnhancer(verbose=args.verbose) as enhancer:
        result = enhancer.process_image(
            image,
            remove_blur=not args.no_deblur,
            add_bokeh=not args.no_bokeh,
            enhance_face=not args.no_face_enhance,
            boost_sharpness=not args.no_sharpness,
            bokeh_intensity=args.bokeh_intensity
        )
    
    # Save result
    cv2.imwrite(output_path, result)
    print(f"✓ Saved to: {output_path}\n")
    
    return True


def process_directory(input_dir, output_dir, args):
    """Process all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Find all images
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in extensions]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    print("=" * 60)
    
    # Process each image
    success_count = 0
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_file.name}")
        
        output_file = output_path / f"enhanced_{img_file.name}"
        
        if process_single_image(str(img_file), str(output_file), args):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✓ Processed {success_count}/{len(image_files)} images successfully")
    print(f"✓ Output directory: {output_dir}")


def create_comparison(input_path, output_path):
    """Create side-by-side comparison image."""
    input_img = cv2.imread(input_path)
    output_img = cv2.imread(output_path)
    
    if input_img is None or output_img is None:
        print("ERROR: Could not read images for comparison")
        return None
    
    # Resize to same height
    h1, w1 = input_img.shape[:2]
    h2, w2 = output_img.shape[:2]
    
    if h1 != h2:
        target_height = min(h1, h2)
        input_img = cv2.resize(input_img, (int(w1 * target_height / h1), target_height))
        output_img = cv2.resize(output_img, (int(w2 * target_height / h2), target_height))
    
    # Add labels
    input_labeled = input_img.copy()
    output_labeled = output_img.copy()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(input_labeled, "BEFORE", (20, 50), font, 1.5, (0, 0, 255), 3)
    cv2.putText(output_labeled, "AFTER", (20, 50), font, 1.5, (0, 255, 0), 3)
    
    # Combine side by side
    comparison = cv2.hconcat([input_labeled, output_labeled])
    
    return comparison


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Studio-Quality Portrait Converter - Transform raw portraits into studio-quality images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  python main.py input.jpg -o output.jpg
  
  # Process all images in a directory
  python main.py input_images/ -o output_images/ --batch
  
  # Process with custom settings
  python main.py portrait.jpg -o enhanced.jpg --bokeh-intensity strong
  
  # Create before/after comparison
  python main.py input.jpg -o output.jpg --compare comparison.jpg
        """
    )
    
    parser.add_argument('input', help='Input image path or directory')
    parser.add_argument('-o', '--output', required=True, 
                       help='Output image path or directory')
    parser.add_argument('--batch', action='store_true',
                       help='Process all images in input directory')
    parser.add_argument('--compare', metavar='PATH',
                       help='Create before/after comparison image at specified path')
    
    # Enhancement options
    parser.add_argument('--no-deblur', action='store_true',
                       help='Skip motion blur removal')
    parser.add_argument('--no-bokeh', action='store_true',
                       help='Skip background blur effect')
    parser.add_argument('--no-face-enhance', action='store_true',
                       help='Skip face enhancement')
    parser.add_argument('--no-sharpness', action='store_true',
                       help='Skip sharpness/contrast enhancement')
    parser.add_argument('--bokeh-intensity', choices=['light', 'medium', 'strong'],
                       default='medium', help='Bokeh effect intensity (default: medium)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed processing information')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"ERROR: Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Process based on mode
    if args.batch:
        if not os.path.isdir(args.input):
            print("ERROR: --batch requires input to be a directory")
            sys.exit(1)
        process_directory(args.input, args.output, args)
    else:
        if os.path.isdir(args.input):
            print("ERROR: Input is a directory. Use --batch flag for batch processing")
            sys.exit(1)
        
        # Process single image
        success = process_single_image(args.input, args.output, args)
        
        if not success:
            sys.exit(1)
        
        # Create comparison if requested
        if args.compare:
            print("\nCreating comparison image...")
            comparison = create_comparison(args.input, args.output)
            if comparison is not None:
                cv2.imwrite(args.compare, comparison)
                print(f"✓ Comparison saved to: {args.compare}")


if __name__ == '__main__':
    main()
