import os
from PIL import Image
import numpy as np

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  

def calculate_average_dimensions(root_dir):
    widths = []
    heights = []
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
                image_path = os.path.join(root, file)
                width, height = get_image_size(image_path)
                widths.append(width)
                heights.append(height)

    if widths and heights:
        average_width = np.mean(widths)
        average_height = np.mean(heights)
        return average_width, average_height, len(heights)
    else:
        return None, None

root_directory = "./benchmarks/MileBench/data/MLBench"
average_width, average_height, num = calculate_average_dimensions(root_directory)

if average_width and average_height:
    print(f"Num: {num}")
    print(f"Average Width: {average_width}")
    print(f"Average Height: {average_height}")
else:
    print("No images found.")
