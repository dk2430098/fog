#!/usr/bin/env python3
"""
Demo Script for Studio-Quality Portrait Converter
Creates demo materials and videos showcasing the enhancement results.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import time

from portrait_enhancer import PortraitEnhancer


def create_comparison_grid(input_images, output_images, save_path, grid_cols=2):
    """
    Create a grid of before/after comparisons.
    
    Args:
        input_images: List of input image paths
        output_images: List of output image paths
        save_path: Path to save the grid
        grid_cols: Number of columns in grid
    """
    pairs = []
    target_width = 400
    
    for inp, out in zip(input_images, output_images):
        img_in = cv2.imread(inp)
        img_out = cv2.imread(out)
        
        if img_in is None or img_out is None:
            continue
        
        # Resize to target width
        h, w = img_in.shape[:2]
        new_h = int(h * target_width / w)
        img_in = cv2.resize(img_in, (target_width, new_h))
        img_out = cv2.resize(img_out, (target_width, new_h))
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_in, "BEFORE", (10, 30), font, 0.8, (0, 0, 255), 2)
        cv2.putText(img_out, "AFTER", (10, 30), font, 0.8, (0, 255, 0), 2)
        
        # Combine pair
        pair = cv2.hconcat([img_in, img_out])
        pairs.append(pair)
    
    if not pairs:
        print("No valid image pairs found")
        return None
    
    # Create grid
    rows = []
    for i in range(0, len(pairs), grid_cols):
        row_pairs = pairs[i:i+grid_cols]
        
        # Pad if needed
        if len(row_pairs) < grid_cols:
            # Create blank image
            blank = np.zeros_like(row_pairs[0])
            row_pairs.extend([blank] * (grid_cols - len(row_pairs)))
        
        row = cv2.hconcat(row_pairs)
        rows.append(row)
    
    grid = cv2.vconcat(rows)
    cv2.imwrite(save_path, grid)
    print(f"✓ Comparison grid saved to: {save_path}")
    
    return grid


def create_demo_video(input_images, output_images, video_path, fps=1):
    """
    Create a demo video showing before/after transitions.
    
    Args:
        input_images: List of input image paths
        output_images: List of output image paths
        video_path: Path to save video
        fps: Frames per second
    """
    if not input_images or not output_images:
        print("No images provided for video")
        return False
    
    # Read first image to get dimensions
    first_in = cv2.imread(input_images[0])
    first_out = cv2.imread(output_images[0])
    
    if first_in is None or first_out is None:
        print("Could not read first image pair")
        return False
    
    # Target dimensions
    target_width = 1280
    h, w = first_in.shape[:2]
    target_height = int(h * target_width / w)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (target_width * 2, target_height))
    
    for inp, outp in zip(input_images, output_images):
        img_in = cv2.imread(inp)
        img_out = cv2.imread(outp)
        
        if img_in is None or img_out is None:
            continue
        
        # Resize
        img_in = cv2.resize(img_in, (target_width, target_height))
        img_out = cv2.resize(img_out, (target_width, target_height))
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_in, "BEFORE", (20, 60), font, 2, (0, 0, 255), 4)
        cv2.putText(img_out, "AFTER", (20, 60), font, 2, (0, 255, 0), 4)
        
        # Combine side by side
        frame = cv2.hconcat([img_in, img_out])
        
        # Write frame (hold for specified fps)
        for _ in range(int(fps * 2)):  # Hold each pair for 2 seconds
            out.write(frame)
    
    out.release()
    print(f"✓ Demo video saved to: {video_path}")
    
    return True


def benchmark_performance(image_path, num_runs=5):
    """
    Benchmark processing performance.
    
    Args:
        image_path: Path to test image
        num_runs: Number of runs for averaging
    """
    print(f"\nBenchmarking performance with {num_runs} runs...")
    print("=" * 60)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image: {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    times = []
    
    with PortraitEnhancer(verbose=False) as enhancer:
        # Warm-up run
        _ = enhancer.process_image(image)
        
        # Benchmark runs
        for i in range(num_runs):
            start = time.time()
            _ = enhancer.process_image(image)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"Run {i+1}/{num_runs}: {elapsed:.3f}s")
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print("\n" + "=" * 60)
    print("Performance Summary:")
    print(f"  Average time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"  Min time: {min_time:.3f}s")
    print(f"  Max time: {max_time:.3f}s")
    print(f"  Throughput: {1/avg_time:.2f} images/second")
    print("=" * 60)


def generate_demo_materials(input_dir='input_images', 
                           output_dir='output_images',
                           demo_dir='demo_outputs'):
    """
    Generate all demo materials.
    
    Args:
        input_dir: Directory with input images
        output_dir: Directory for enhanced outputs
        demo_dir: Directory for demo materials
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    demo_path = Path(demo_dir)
    
    # Create directories
    output_path.mkdir(parents=True, exist_ok=True)
    demo_path.mkdir(parents=True, exist_ok=True)
    
    # Find images
    extensions = {'.jpg', '.jpeg', '.png'}
    input_images = [str(f) for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in extensions]
    
    if not input_images:
        print(f"No images found in {input_dir}")
        return
    
    print(f"\nGenerating demo materials for {len(input_images)} images...")
    print("=" * 60)
    
    # Process all images
    output_images = []
    
    with PortraitEnhancer(verbose=True) as enhancer:
        for i, img_path in enumerate(input_images, 1):
            print(f"\n[{i}/{len(input_images)}] Processing: {Path(img_path).name}")
            
            # Read and process
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            result = enhancer.process_image(image)
            
            # Save output
            output_file = output_path / f"enhanced_{Path(img_path).name}"
            cv2.imwrite(str(output_file), result)
            output_images.append(str(output_file))
            
            # Create individual comparison
            comparison_file = demo_path / f"comparison_{Path(img_path).name}"
            
            # Create side-by-side
            img_in = cv2.imread(img_path)
            h, w = img_in.shape[:2]
            target_w = 800
            target_h = int(h * target_w / w)
            
            img_in = cv2.resize(img_in, (target_w, target_h))
            img_out = cv2.resize(result, (target_w, target_h))
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_in, "BEFORE", (20, 50), font, 1.5, (0, 0, 255), 3)
            cv2.putText(img_out, "AFTER", (20, 50), font, 1.5, (0, 255, 0), 3)
            
            comparison = cv2.hconcat([img_in, img_out])
            cv2.imwrite(str(comparison_file), comparison)
    
    # Create comparison grid
    if len(input_images) > 1:
        grid_file = demo_path / "comparison_grid.jpg"
        create_comparison_grid(input_images, output_images, str(grid_file))
    
    # Create demo video
    if len(input_images) > 0:
        video_file = demo_path / "demo_video.mp4"
        create_demo_video(input_images, output_images, str(video_file), fps=1)
    
    print("\n" + "=" * 60)
    print("✓ Demo materials generated successfully!")
    print(f"✓ Enhanced images: {output_dir}/")
    print(f"✓ Demo materials: {demo_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate demo materials for Portrait Enhancer')
    parser.add_argument('--input', default='input_images', help='Input directory')
    parser.add_argument('--output', default='output_images', help='Output directory')
    parser.add_argument('--demo', default='demo_outputs', help='Demo materials directory')
    parser.add_argument('--benchmark', metavar='IMAGE', help='Run performance benchmark on image')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_performance(args.benchmark)
    else:
        generate_demo_materials(args.input, args.output, args.demo)
