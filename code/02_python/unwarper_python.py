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
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z], dtype=np.float64)

def get_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Generate rotation matrix from yaw, pitch, roll angles (in degrees).
    
    Args:
        yaw: Rotation around Y axis in degrees
        pitch: Rotation around X axis in degrees
        roll: Rotation around Z axis in degrees
    Returns:
        3x3 rotation matrix as numpy array
    """
    # Yaw quaternion, rotate view around Y axis
    yaw = np.deg2rad(0)
    yaw_q = np.array([np.cos(yaw/2.0), 0.0, np.sin(yaw/2.0), 0.0], dtype=np.float64)
    # Pitch quaternion, rotate view around X axis (look up 45 degrees)
    pitch = np.deg2rad(45)             
    pitch_q = np.array([np.cos(pitch/2.0), np.sin(pitch/2.0), 0.0, 0.0], dtype=np.float64)
    # Roll quaternion, rotate view around Z axis (look in different direction)
    roll = np.deg2rad(roll)
    roll_q = np.array([np.cos(roll/2.0), 0.0, 0.0, np.sin(roll/2.0)], dtype=np.float64)

    rq = multiply_quaternion(roll_q, multiply_quaternion(pitch_q, yaw_q))

    # Build spherical projection matrix from quaternions
    w, x, y, z = rq
    return np.array([
        [ (w*w + x*x - y*y - z*z),  2.0 * (x*y - z*w), 2.0 * (w*y + x*z)],
        [ 2.0 * (w*z + x*y), (w*w - x*x + y*y - z*z), 2.0 * (y*z - w*x)],
        [ 2.0 * (x*z - y*w), 2.0 * (w*x + y*z), (w*w - x*x - y*y + z*z)]], dtype=np.float64)


def project_pixel(xyz: np.ndarray, width: int, height: int) -> Tuple[int, int]:
    hs = np.hypot(xyz[0],xyz[1])
    phi = np.arctan2(hs, xyz[2])
    coeff = phi / (hs * np.pi)
    src_x = xyz[0] * coeff + 0.5
    src_y = xyz[1] * coeff + 0.5
    return src_x, src_y


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
       
        remap = []
        for zone_id in range(self.zones):
    
            # Get rotation matrix for this zone
            R = get_rotation_matrix(0, 45, zone_id * (360.0 / self.zones))

            remap_zone = []
            for j in range(self.output_height):
                line = []
                for i in range(self.output_width):
                    v = np.array([i / (0.5 * self.output_width) - 1.0, j / (0.5 * self.output_height) - 1.0, 1.0])
                    xyz = R @ v.T
                    src_x, src_y = project_pixel(xyz, self.width, self.height)
                    map_y = int(src_y * self.height)
                    map_x = int(src_x * self.width)
                    if 0 <= map_y < self.height and 0 <= map_x < self.width:    
                        line.append((map_y, map_x))
                    else:
                        line.append((0, 0))
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
                # Note: never out of bound as it is ensured when building remapping
                output_buffer[i, j] = image[remap_table[i][j]]

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
