#!/usr/bin/env python3
"""
Fisheye Unwarper using ctypes and C++ Library

A Python script that uses ctypes to call a C++ library for fisheye dewarping.
This implementation combines the numpy dewarping mapping generation with C++
execution for better performance.
"""

import numpy as np
import argparse
import os
import sys
import ctypes
from ctypes import c_int, c_uint8, POINTER, c_void_p
from PIL import Image
import math
from typing import Tuple, Optional

# Load the C++ library
def load_library():
    """Load the C++ library from unwarper_ctypes directory."""
    lib_path = os.path.join(os.path.dirname(__file__), 'unwarper_ctypes', 'libunwarper_ctypes.so')
    
    # Try to load from current directory first
    try:
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from: {lib_path}")
        return lib
    except OSError as e:
        # Try loading with LD_LIBRARY_PATH
        try:
            lib = ctypes.CDLL('libunwarper_ctypes.so')
            print("Loaded library using LD_LIBRARY_PATH")
            return lib
        except OSError:
            print(f"Error: Could not load C++ library from {lib_path}")
            print("Make sure the library is built and LD_LIBRARY_PATH is set correctly:")
            print(f"  cd unwarper_ctypes && make")
            print(f"  export LD_LIBRARY_PATH={os.path.dirname(os.path.abspath(lib_path))}:$LD_LIBRARY_PATH")
            sys.exit(1)

# Load and configure the library
lib = load_library()

# Define function signatures
lib.create_dewarp_context.argtypes = [c_int, c_int, c_int]
lib.create_dewarp_context.restype = c_void_p

lib.get_width.argtypes = [c_void_p]
lib.get_width.restype = c_int

lib.get_height.argtypes = [c_void_p]
lib.get_height.restype = c_int

lib.get_output_width.argtypes = [c_void_p]
lib.get_output_width.restype = c_int

lib.get_output_height.argtypes = [c_void_p]
lib.get_output_height.restype = c_int

lib.get_zones.argtypes = [c_void_p]
lib.get_zones.restype = c_int

lib.dewarp_frame.argtypes = [c_void_p, POINTER(c_uint8), POINTER(c_uint8), c_int]
lib.dewarp_frame.restype = None

lib.free_dewarp_context.argtypes = [c_void_p]
lib.free_dewarp_context.restype = None


class CTypesDewarper:
    """
    C++ accelerated dewarper using ctypes interface.
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
        self.zones = zones
        
        # Create C++ dewarp context
        self.ctx = lib.create_dewarp_context(width, height, zones)
        if not self.ctx:
            raise RuntimeError("Failed to create dewarp context")
        
        self.output_width = lib.get_output_width(self.ctx)
        self.output_height = lib.get_output_height(self.ctx)
        # Create output buffer
        self.output_buffer = np.zeros((self.zones, self.output_height, self.output_width, 3), dtype=np.uint8)

    
    def dewarp_frame(self, image: np.ndarray, zone_id: int = -1):
        """
        Apply dewarping transformation to image using C++ library.
        
        Args:
            image: Input image as NumPy array (H, W, 3)
            zone_id: Zone ID to use for dewarping
            
        Returns:
            Dewarped image as NumPy array
        """
        if zone_id < 0 or zone_id >= self.zones:
            raise ValueError(f"Invalid zone_id: {zone_id}. Must be 0-{self.zones-1}")
        
        # Ensure image is contiguous and in the right format
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
                
        # Get ctypes pointers
        input_ptr = image.ctypes.data_as(POINTER(c_uint8))
        output_ptr = self.output_buffer[zone_id].ctypes.data_as(POINTER(c_uint8))
        
        # Call C++ function
        lib.dewarp_frame(self.ctx, input_ptr, output_ptr, zone_id)
        
        return self.output_buffer[zone_id]
    
    def __del__(self):
        """Clean up C++ context."""
        if hasattr(self, 'ctx') and self.ctx:
            lib.free_dewarp_context(self.ctx)


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
        description='Fisheye Unwarper (ctypes + C++) - Convert fisheye images to multiple perspective views'
    )
    parser.add_argument('input_image', help='Path to input fisheye image')
    parser.add_argument('--output-dir', default='.', help='Output directory (default: current directory)')
    parser.add_argument('--prefix', help='Output filename prefix (default: input filename)')
    parser.add_argument('--zones', type=int, default=5, help='Number of perspective views (default: 5)')
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
    print(f"Initializing C++ dewarper for {width}x{height} image with {args.zones} zones")
    try:
        dewarper = CTypesDewarper(width, height, args.zones)
    except Exception as e:
        print(f"Error initializing dewarper: {e}")
        sys.exit(1)
    
    output_basename = args.prefix or os.path.splitext(os.path.basename(args.input_image))[0]

    # Process each zone
    for zone_id in range(dewarper.zones):
        print(f"Processing zone {zone_id + 1}/{dewarper.zones}...")
        zone_image = None
        for _ in range(args.repeat_dewarp):
            zone_image = dewarper.dewarp_frame(image, zone_id)
        # Ensure we have at least one dewarped image
        if zone_image is None:
            zone_image = dewarper.dewarp_frame(image, zone_id)
        output_path = os.path.join(args.output_dir, f"{output_basename}_{zone_id + 1}_ctypes_opt.jpg")
        save_image(zone_image, output_path)
        print(f"Saved view {zone_id + 1} to: {output_path}")
        
    print("Dewarping complete!")
    print(f"Generated {dewarper.zones} perspective views using C++ acceleration")


if __name__ == '__main__':
    main()