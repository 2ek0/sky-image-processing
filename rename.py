import os
import re

# Define the folder containing the images
folder_path = r"D:\Image"

# Regex pattern to match the filename format
pattern = re.compile(r"201D_CAM1_(\d{8}_\d{6})_\d{2}\.jpg")

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        new_name = match.group(1) + ".jpg"  # Extracts only "YYYYMMDD_HHMMSS.jpg"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")

print("✅ All files renamed successfully!")
