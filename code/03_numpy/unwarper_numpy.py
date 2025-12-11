#!/usr/bin/env python3
"""
Pure Python Fisheye Unwarper

A standalone Python script that replicates the functionality of the C++ fisheye 
dewarping library using pure Python with NumPy.

This implementation ports all quaternion mathematics and dewarping algorithms
from the original C++ code to maintain identical output.
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
        
        # Create remapping tables for all zones
        self.remap = self._dewarp_mapping()
        self.output_buffer = np.zeros((self.zones, self.output_height, self.output_width, 3), dtype=np.uint8)

    
    def get_rotation_matrix(self, yaw: float, pitch: float, roll: float) -> np.ndarray:
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

    def _dewarp_mapping(self):
        """
        Create pixel remapping tables for all zones.
        
        This is the core dewarping algorithm using spherical projection.
        Returns a numpy array with shape (zones, output_height, output_width, 2)
        where the last dimension contains (src_x, src_y) for each pixel.
        """
        
        # Create remapping tables for each view and stack them into a single array
        # Shape: (zones, output_height, output_width, 2)
        # Last dimension contains (src_x, src_y) for each pixel
        remap_list = []
        # These parameters control the focal length and centering of the projection
        expand = 1.269 # Experimental expansion factor to cover full frame (match ffmpeg output)
        offset = 0.25 # Experimental offset to center the projection (matching ffmpeg)
        v = np.array([expand * 4.0 / self.width, expand * 4.0 / self.height, 4.0 / np.pi], dtype=np.float64)  # Up vector
        for zone_id in range(self.zones):

            R = self.get_rotation_matrix(0.0, 45.0, 360.0 * zone_id / self.zones)
            m = R * v # Rotation matrix scaled by view parameters
            # Build remapping table using vectorized operations
            offset_width = offset * self.width
            offset_height = offset * self.height

            j_coords, i_coords = np.meshgrid(
                np.arange(self.output_width),
                np.arange(self.output_height), indexing='ij')
            x_coords = i_coords.flatten() - offset_width
            y_coords = j_coords.flatten() - offset_height

            coords = np.column_stack([x_coords, y_coords, np.ones_like(x_coords)])

            xyz = coords @ m.T
            hs = np.hypot(xyz[:, 0], xyz[:, 1])
            phi = np.arctan2(hs, xyz[:, 2])

            # Vectorized source coordinate calculation, integer coordinates
            src_x = (self.width * (xyz[:, 0] * phi / (np.pi * hs) + 0.5)).astype(np.int32)
            src_y = (self.height * (xyz[:, 1] * phi / (np.pi * hs) + 0.5)).astype(np.int32)

            # Clip to valid range
            src_x_int = np.clip(src_x, 0, self.width - 1)
            src_y_int = np.clip(src_y, 0, self.height - 1)

            # Reshape to 2D arrays
            src_x_2d = src_x_int.reshape((self.output_height, self.output_width))
            src_y_2d = src_y_int.reshape((self.output_height, self.output_width))
            
            # Stack to create zone mapping with shape (output_height, output_width, 2)
            zone_mapping = np.stack([src_x_2d, src_y_2d], axis=-1)
            remap_list.append(zone_mapping)
        
        # Stack all zones together with shape (zones, output_height, output_width, 2)
        return np.stack(remap_list, axis=0)
    
    def dewarp_frame(self, image: np.ndarray, zone_id: int = -1):
        """
        Apply dewarping transformation to image.
        
        Args:
            image: Input image as NumPy array (H, W, 3)
            zone_id: Zone ID to use for dewarping
        """
        
        # Get the mapping for the specified zone
        # Shape: (output_height, output_width, 2)
        remap_table = self.remap[zone_id]
        
        # Extract source coordinates
        src_x = remap_table[:, :, 0]
        src_y = remap_table[:, :, 1]
        
        # Initialize output buffer
        output_buffer = self.output_buffer[zone_id]
        
        # Apply vectorized mapping using advanced indexing
        output_buffer = image[src_y, src_x]
        
        # Invalid pixels remain black (already initialized as zeros)

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
        zone_image = dewarper.dewarp_frame(image, zone_id)
        # Repeat dewarping if requested (for performance testing)
        for _ in range(args.repeat_dewarp - 1):
            zone_image = dewarper.dewarp_frame(image, zone_id)
        output_path = os.path.join(args.output_dir, f"{output_basename}_{zone_id + 1}_numpy.jpg")
        save_image(zone_image, output_path)
        print(f"Saved view {zone_id + 1} to: {output_path}")
    
    print("Dewarping complete!")


if __name__ == '__main__':
    main()
