#!/usr/bin/env python3
"""
Pure Python Fisheye Unwarper

Cette implémentation en Python pur du dewarping fisheye utilise uniquement
des bibliothèques standard et numpy pour manipuler des tableaux,
sans aucune optimisation vectorielle.

Elle sert de référence pour comparer les performances avec d'autres implémentations
optimisées en C++, ctypes, OpenCV ou NumPy.
"""

import numpy as np
import argparse
import os
import sys
from PIL import Image
import math
from typing import Tuple, Optional, List


def multiply_quaternion(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions using Python primitives.
    
    Args:
        a: First quaternion [w, x, y, z]
        b: Second quaternion [w, x, y, z]
        
    Returns:
        Result quaternion [w, x, y, z]
    """
    w1, x1, y1, z1 = a[0], a[1], a[2], a[3]
    w2, x2, y2, z2 = b[0], b[1], b[2], b[3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z], dtype=np.float64)


class PythonDewarper:
    """
    Pure Python implementation of the fisheye dewarper.    
    """
    
    def __init__(self, width: int, height: int, zones: int = 3):
        """
        Initialize dewarper with image dimensions.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
        """
        self.width = width
        self.height = height
        self.output_width = self.width // 2
        self.output_height = self.height // 2
        self.zones = zones
        
        # Remapping tables for each view
        self.remap = self._dewarp_mapping()
        self.output_buffer = np.zeros((self.zones, self.output_height, self.output_width, 3), dtype=np.uint8)
    
    def _dewarp_mapping(self) -> List[List[List[Tuple[int, int]]]]:
        """
        Create pixel remapping table for specific view.
        
        This is the core dewarping algorithm using spherical projection.
        """
       
        # Yaw quaternion, rotate view around Y axis
        yaw = 0 * np.pi / 360.0
        sin_yaw, cos_yaw = np.sin(yaw), np.cos(yaw)
        yaw_q = np.array([cos_yaw, 0.0, sin_yaw, 0.0], dtype=np.float64)
        # Pitch quaternion, rotate view around X axis (look up 45 degrees)
        pitch = 45 * np.pi / 360.0             
        sin_pitch, cos_pitch = np.sin(pitch), np.cos(pitch)
        pitch_q = np.array([cos_pitch, sin_pitch, 0.0, 0.0], dtype=np.float64)

        remap = []
        for zone_id in range(self.zones):
            # Roll quaternion, rotate view around Z axis (look in different direction)
            roll = (360.0 * zone_id / self.zones) * np.pi / 360.0
            sin_roll, cos_roll = np.sin(roll), np.cos(roll)
            roll_q = np.array([cos_roll, 0.0, 0.0, sin_roll], dtype=np.float64)
    
            rq = multiply_quaternion(multiply_quaternion(roll_q, pitch_q), yaw_q)
            
            expand = 1.269 # Experimental expansion factor to cover full frame (match ffmpeg output)
            offset = 0.25 # Experimental offset to center the projection (matching ffmpeg)
         
            # Build spherical projection matrix from quaternions
            m = np.array([[
                    expand * 4.0 * (rq[0]**2 + rq[1]**2 - rq[2]**2 - rq[3]**2) / self.width,
                    expand * 4.0 * (-rq[0] * rq[3] + rq[1] * rq[2] + rq[2] * rq[1] - rq[3] * rq[0]) / self.height,
                    4.0 * (rq[0] * rq[2] + rq[1] * rq[3] + rq[2] * rq[0] + rq[3] * rq[1]) / np.pi
                ],
                [
                    expand * 4.0 * (rq[0] * rq[3] + rq[1] * rq[2] + rq[2] * rq[1] + rq[3] * rq[0]) / self.width,
                    expand * 4.0 * (rq[0]**2 - rq[1]**2 + rq[2]**2 - rq[3]**2) / self.height,
                    4.0 * (-rq[0] * rq[1] - rq[1] * rq[0] + rq[2] * rq[3] + rq[3] * rq[2])  / np.pi
                ],
                [
                    expand * 4.0 * (-rq[0] * rq[2] + rq[1] * rq[3] - rq[2] * rq[0] + rq[3] * rq[1]) / self.width,
                    expand * 4.0 * (rq[0] * rq[1] + rq[1] * rq[0] + rq[2] * rq[3] + rq[3] * rq[2]) / self.height,
                    4.0 * (rq[0]**2 - rq[1]**2 - rq[2]**2 + rq[3]**2)  / np.pi
                ]], dtype=np.float64)

            # Build remapping table
            offset_width = offset * self.width
            offset_height = offset * self.height

            remap_zone = []
            for j in range(self.output_height):
                line = []
                y = j - offset_height
                for i in range(self.output_width):
                    x = i - offset_width               
                    xyz = ((m[0, 0] * x + m[0, 1] * y + m[0, 2]),
                        (m[1, 0] * x + m[1, 1] * y + m[1, 2]),
                        (m[2, 0] * x + m[2, 1] * y + m[2, 2]))
                    
                    hs = np.hypot(xyz[0], xyz[1])
                    phi = np.arctan2(hs, xyz[2])
                    src_x = int(self.width * (xyz[0] * phi / (np.pi * hs) + 0.5))
                    src_y = int(self.height * (xyz[1] * phi / (np.pi * hs) + 0.5))
                    line.append((src_y, src_x))
                remap_zone.append(line)
            remap.append(remap_zone)

        return remap
    
    def dewarp_frame(self, image: np.ndarray, zone_id: int = -1):
        """
        Apply dewarping transformation to image.
        
        Args:
            image: Input image as NumPy array (H, W, 3)
        """
        
        remap_table = self.remap[zone_id]
        output_buffer = self.output_buffer[zone_id]

        for i in range(self.output_height):
            for j in range(self.output_width):
                try:
                    output_buffer[i, j] = image[remap_table[i][j]]
                except IndexError:
                    output_buffer[i, j] = [0, 0, 0]  # Black for out-of-bounds

        return output_buffer.reshape((self.output_height, self.output_width, 3))

def load_image(image_path: str) -> np.ndarray:
    """Load image and convert to RGB NumPy array."""
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        sys.exit(1)


def save_image(image: np.ndarray, output_path: str):
    """Save NumPy array as image."""
    try:
        Image.fromarray(image).save(output_path)
    except Exception as e:
        print(f"Error saving image {output_path}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Pure Python Fisheye Unwarper - Convert fisheye images to 3 perspective views'
    )
    parser.add_argument('input_image', help='Path to input fisheye image')
    parser.add_argument('--output-dir', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--prefix', help='Output filename prefix (default: input filename)')
    parser.add_argument('--repeat-dewarp', '-r', help='Number of repetition of dewarping for performance testing', type=int, default=1)
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_image):
        print(f"Error: Input file '{args.input_image}' does not exist")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load input image
    print(f"Loading image: {args.input_image}")
    image = load_image(args.input_image)
    height, width = image.shape[:2]
    
    # Initialize dewarper
    print(f"Initializing dewarper for {width}x{height} image")
    dewarper = PythonDewarper(width, height, 5)         
    output_basename = args.prefix or os.path.splitext(os.path.basename(args.input_image))[0]
    for zone_id in range(dewarper.zones):
        zone_image = None
        for _ in range(args.repeat_dewarp):
            zone_image = dewarper.dewarp_frame(image, zone_id)
        # Ensure we have at least one dewarped image
        if zone_image is None:
            zone_image = dewarper.dewarp_frame(image, zone_id)
        output_path = os.path.join(args.output_dir, f"{output_basename}_{zone_id + 1}_python.jpg")
        save_image(zone_image, output_path)
        print(f"Saved view {zone_id + 1} to: {output_path}")
    
    print("Dewarping complete!")


if __name__ == '__main__':
    main()
