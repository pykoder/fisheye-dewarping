#!/usr/bin/env python3
"""
Fisheye Composite Image Generator

Creates an infographic showing a fisheye image divided into 5 sectors of 72°
with red dividing lines and corresponding dewarped images at the end of each line.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import math
import sys


def create_composite(fisheye_path, dewarped_paths, output_path):
    """
    Create composite infographic.
    
    Args:
        fisheye_path: Path to fisheye image
        dewarped_paths: List of 5 paths to dewarped images
        output_path: Output path for composite PNG
    """
    
    # Load fisheye image
    fisheye = Image.open(fisheye_path).convert('RGB')
    fisheye_width, fisheye_height = fisheye.size
    
    # Load dewarped images
    dewarped_images = [Image.open(path).convert('RGB') for path in dewarped_paths]
    dewarped_width, dewarped_height = dewarped_images[0].size
    
    # Calculate canvas size
    # Need space for fisheye + dewarped images radiating outward
    # Line extension beyond fisheye
    line_extension = 250
    # Distance from fisheye center to dewarped image center
    dewarped_distance = fisheye_width // 2 + line_extension + dewarped_width // 2
    
    # Canvas size to fit everything
    canvas_size = int(fisheye_width + 2 * dewarped_distance + 200)
    
    # Create canvas
    canvas = Image.new('RGB', (canvas_size, canvas_size), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Center coordinates
    center_x = canvas_size // 2
    center_y = canvas_size // 2
    
    # Paste fisheye at center
    fisheye_x = center_x - fisheye_width // 2
    fisheye_y = center_y - fisheye_height // 2
    canvas.paste(fisheye, (fisheye_x, fisheye_y))
    
    # Fisheye radius (assuming square image)
    fisheye_radius = fisheye_width // 2
    
    offset_x = [0, 0, 40, -40, 0]
    offset_y = [80, 0, 80, 80, 0]
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            font = ImageFont.load_default()
    
    # Draw sectors and place dewarped images
    for sector_id in range(5):
        # Angle for this sector (starting from vertical = -90°, going clockwise)
        # Sector 1 starts at vertical (top), which is -90° in standard coords
        angle_deg = -90 + (sector_id * 72)
        angle_rad = math.radians(angle_deg)
        
        # Calculate line endpoints
        # Line starts from fisheye edge
        start_x = center_x + int(fisheye_radius * math.cos(angle_rad))
        start_y = center_y + int(fisheye_radius * math.sin(angle_rad))
        
        # Line extends beyond fisheye
        end_x = center_x + int((fisheye_radius + line_extension) * math.cos(angle_rad))
        end_y = center_y + int((fisheye_radius + line_extension) * math.sin(angle_rad))
        
        # Draw red line
        draw.line([(center_x, center_y), (end_x, end_y)], fill='red', width=3)
        
        # Position for dewarped image (centered on extended line endpoint)
        dewarped_center_x = center_x + int(dewarped_distance * math.cos(angle_rad))
        dewarped_center_y = center_y + int(dewarped_distance * math.sin(angle_rad))
        
        dewarped_x = dewarped_center_x - dewarped_width // 2
        dewarped_y = dewarped_center_y - dewarped_height // 2
        
        # Paste dewarped image
        canvas.paste(dewarped_images[sector_id], (dewarped_x+offset_x[sector_id], dewarped_y+offset_y[sector_id]))
        
        # Draw border around dewarped image
        draw.rectangle(
            [(dewarped_x +offset_x[sector_id] , dewarped_y +offset_y[sector_id]), 
             (dewarped_x + dewarped_width +offset_x[sector_id], dewarped_y + dewarped_height +offset_y[sector_id] )],
            outline='red',
            width=2
        )
        
        # Add sector label
        label = f"Secteur {sector_id + 1}"
        # Position label near dewarped image
        label_x = dewarped_center_x +offset_x[sector_id]
        label_y = dewarped_y - 20 + +offset_y[sector_id]
        draw.text((label_x, label_y), label, fill='red', anchor='mm', font=font)
    
    # Draw circle around fisheye to show boundary
    draw.ellipse(
        [(center_x - fisheye_radius, center_y - fisheye_radius),
         (center_x + fisheye_radius, center_y + fisheye_radius)],
        outline='red',
        width=2
    )
    
    # Save composite
    canvas.save(output_path, 'PNG')
    print(f"Composite saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate fisheye composite infographic'
    )
    parser.add_argument('fisheye', help='Path to fisheye image')
    parser.add_argument('dewarped', nargs=5, help='Paths to 5 dewarped images (in order)')
    parser.add_argument('--output', '-o', default='composite.png', 
                       help='Output path (default: composite.png)')
    
    args = parser.parse_args()
    
    create_composite(args.fisheye, args.dewarped, args.output)


if __name__ == '__main__':
    main()
