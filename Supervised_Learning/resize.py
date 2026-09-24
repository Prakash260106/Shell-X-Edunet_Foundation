from PIL import Image
import glob
import os

images = glob.glob("assets/*.png")
for img_path in images:
    with Image.open(img_path) as img:
        # We want to crop/resize it to be wider, like 800x400
        # For a center crop:
        target_width = 800
        target_height = 400
        
        # Calculate aspect ratios
        img_ratio = img.width / img.height
        target_ratio = target_width / target_height
        
        if img_ratio > target_ratio:
            # Image is wider than target ratio
            new_width = int(target_ratio * img.height)
            offset = (img.width - new_width) / 2
            crop_box = (offset, 0, img.width - offset, img.height)
        else:
            # Image is taller than target ratio
            new_height = int(img.width / target_ratio)
            offset = (img.height - new_height) / 2
            crop_box = (0, offset, img.width, img.height - offset)
            
        img_cropped = img.crop(crop_box)
        img_resized = img_cropped.resize((target_width, target_height), Image.LANCZOS)
        img_resized.save(img_path)
        print(f"Resized {img_path}")
