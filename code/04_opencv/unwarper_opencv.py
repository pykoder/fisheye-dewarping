#!/usr/bin/env python3
"""
OpenCV Fisheye Unwarper

A Python script that uses OpenCV's fisheye features to dewarp fisheye images
into multiple perspective views, maintaining compatibility with the other
unwarper implementations.
"""

import cv2
import numpy as np
import argparse
import os
import sys
from typing import Tuple, Optional


class OpenCVDewarper:
    """
    OpenCV implementation of the fisheye dewarper using OpenCV's fisheye module
    and custom remapping functions.
    """
    
    def __init__(self, width: int, height: int, zones: int = 3):
        """
        Initialize dewarper with image dimensions.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            zones: Number of perspective views to generate
        """
        self.width = width
        self.height = height
        self.output_width = self.width // 2
        self.output_height = self.height // 2
        self.zones = zones
               
        # Create remapping tables for all zones
        self.remap = self._dewarp_mapping()
    
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
        # Create rotation matrix from Euler angles
        # Convert to radians
        yaw_rad = np.deg2rad(yaw)
        pitch_rad = np.deg2rad(pitch)
        roll_rad = np.deg2rad(roll)
        
        # Calculate rotation matrices
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ], dtype=np.float32)
        
        R_y = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ], dtype=np.float32)
        
        R_z = np.array([
            [np.cos(roll_rad), -np.sin(roll_rad), 0],
            [np.sin(roll_rad), np.cos(roll_rad), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Combined rotation matrix
        R = R_z @ R_y @ R_x
        return R


    def _dewarp_mapping(self):
        """
        Create pixel remapping tables for all zones using OpenCV's fisheye model.
        
        Returns a numpy array with shape (zones, output_height, output_width, 2)
        where the last dimension contains (src_x, src_y) for each pixel.
        """
        # Create remapping tables for each view and stack them into a single array
        remap_list = []
        
        for zone_id in range(self.zones):
            
            R = self.get_rotation_matrix(0, 45, (360.0 * zone_id / self.zones))

            # New camera matrix for the perspective view
            # Adjust field of view to match the expected output
            fov_scale = 1.0  # OpenCV has same behavior as ffmpeg here
            offset = 0.0     # OpenCV has same behavior as ffmpeg here
            K_new = np.array([
                [self.output_width * fov_scale / 2, 0, self.output_width / 2 + offset * self.output_width],
                [0, self.output_height * fov_scale / 2, self.output_height / 2 + offset * self.output_height],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Generate destination points for the perspective view
            x, y = np.meshgrid(
                np.arange(self.output_width),
                np.arange(self.output_height),
                indexing='xy'
            )
            
            # Convert to homogeneous coordinates
            ones = np.ones(x.shape)
            dst_points = np.stack([x.flatten(), y.flatten(), ones.flatten()], axis=1).T
            
            # Project to 3D ray directions
            rays = np.linalg.inv(K_new) @ dst_points
            
            # Normalize rays
            rays = rays / np.linalg.norm(rays, axis=0, keepdims=True)
            
            # Transform rays to fisheye camera coordinate system
            rays_fisheye = R @ rays
            
            # Project fisheye rays to image coordinates
            # For fisheye, we use spherical projection
            theta = np.arccos(np.clip(rays_fisheye[2, :], -1, 1))  # Angle from optical axis
            phi = np.arctan2(rays_fisheye[1, :], rays_fisheye[0, :])  # Azimuthal angle
            
            # Map to fisheye image coordinates
            # Assuming equidistant fisheye projection model
            r = theta * self.width / np.pi  # Radius in fisheye image
            
            # Convert to Cartesian coordinates
            x = r * np.cos(phi) + self.width / 2
            y = r * np.sin(phi) + self.height / 2
            
            # Clip to valid range
            x = np.clip(x, 0, self.width - 1)
            y = np.clip(y, 0, self.height - 1)
            
            # Reshape to 2D arrays
            src_x = x.reshape((self.output_height, self.output_width)).astype(np.float32)
            src_y = y.reshape((self.output_height, self.output_width)).astype(np.float32)
            
            # Stack to create zone mapping with shape (output_height, output_width, 2)
            zone_mapping = np.stack([src_x, src_y], axis=-1)
            remap_list.append(zone_mapping)
        
        # Stack all zones together with shape (zones, output_height, output_width, 2)
        return np.stack(remap_list, axis=0)
    
    def dewarp_frame(self, image: np.ndarray, zone_id: int = 0):
        """
        Apply dewarping transformation to image using OpenCV's remap function.
        
        Args:
            image: Input image as NumPy array (H, W, 3)
            zone_id: Zone ID to use for dewarping
            
        Returns:
            Dewarped image as NumPy array
        """
        # Get the mapping for the specified zone
        # Shape: (output_height, output_width, 2)
        remap_table = self.remap[zone_id]
        
        # Split into x and y coordinate maps
        map_x = remap_table[:, :, 0].astype(np.float32)
        map_y = remap_table[:, :, 1].astype(np.float32)
        
        # Apply OpenCV's remap function
        # Use INTER_NEAREST for best performances
        # BORDER_CONSTANT with value 0 for black borders
        output = cv2.remap(
            image, 
            map_x, 
            map_y, 
            cv2.INTER_NEAREST, 
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        return output


def load_image(image_path: str) -> np.ndarray:
    """Load image using OpenCV and convert to RGB."""
    try:
        # OpenCV loads images in BGR format by default
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        sys.exit(1)


def save_image(image: np.ndarray, output_path: str):
    """Save image using OpenCV (convert from RGB to BGR)."""
    try:
        # Convert from RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
    except Exception as e:
        print(f"Error saving image {output_path}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='OpenCV Fisheye Unwarper - Convert fisheye images to perspective views using OpenCV'
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
    dewarper = OpenCVDewarper(width, height, 5)
    output_basename = args.prefix or os.path.splitext(os.path.basename(args.input_image))[0]
    
    for zone_id in range(dewarper.zones):
        zone_image = None
        for _ in range(args.repeat_dewarp):
            zone_image = dewarper.dewarp_frame(image, zone_id)
        # Ensure we have at least one dewarped image
        if zone_image is None:
            zone_image = dewarper.dewarp_frame(image, zone_id)
        output_path = os.path.join(args.output_dir, f"{output_basename}_{zone_id + 1}_opencv.jpg")
        save_image(zone_image, output_path)
        print(f"Saved view {zone_id + 1} to: {output_path}")
    
    print("Dewarping complete!")


if __name__ == '__main__':
    main()