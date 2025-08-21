import os
import re
import json
from datetime import datetime
import shutil

# --- CONFIGURATION ---
IMAGE_DIR = r"D:\Image"
JSON_BASE_DIR = r"D:\2025\json"
SOURCE_DIR = r"D:\2025"

# --- STEP 0: COPY FILES FROM SOURCE TO IMAGE DIR ---
def copy_files_to_image_dir():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(IMAGE_DIR, file)
            counter = 1
            while os.path.exists(destination_path):
                file_name, file_ext = os.path.splitext(file)
                destination_path = os.path.join(IMAGE_DIR, f"{file_name}_{counter}{file_ext}")
                counter += 1
            shutil.copy2(source_path, destination_path)
    print(f"✅ All files copied successfully to {IMAGE_DIR}")
# --- STEP 1: RENAME IMAGE FILES ---
def rename_images():
    pattern = re.compile(r"201D_CAM1_(\d{8}_\d{6})_\d{2}\.jpg")
    for filename in os.listdir(IMAGE_DIR):
        match = pattern.match(filename)
        if match:
            new_name = match.group(1) + ".jpg"
            old_path = os.path.join(IMAGE_DIR, filename)
            new_path = os.path.join(IMAGE_DIR, new_name)
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} → {new_name}")
    print("✅ All files renamed successfully!")

# --- STEP 2: EDIT JSON FILES ---
def edit_json_files():
    cutoff1 = datetime(2025, 3, 14, 13, 0, 0)
    cutoff2 = datetime(2025, 4, 28, 10, 40, 0)
    for month in os.listdir(JSON_BASE_DIR):
        month_path = os.path.join(JSON_BASE_DIR, month)
        if os.path.isdir(month_path):
            for date in os.listdir(month_path):
                date_path = os.path.join(month_path, date)
                if os.path.isfile(date_path) and date_path.endswith(".json"):
                    processed_data = []
                    with open(date_path, "r") as f:
                        for line in f:
                            record = json.loads(line.strip())
                            dt = datetime.strptime(record["now"], "%Y%m%d_%H%M%S")
                            if dt < cutoff1:
                                factor = 1500 / 65536
                            elif dt < cutoff2:
                                factor = 1200 / 65536
                            else:
                                factor = 1200 / 32768
                            record["irradiance"] = round(record["irradiance"] * factor, 2)
                            formatted_time = dt.strftime("%Y%m%d_%H%M%S")
                            record["now"] = formatted_time
                            processed_data.append(record)
                    # Overwrite with processed data
                    with open(date_path, "w") as f:
                        for record in processed_data:
                            f.write(json.dumps(record) + "\n")
                    print(f"✅ Edited: {date_path}")
    print("🎉 All JSON files edited!")

# --- STEP 3: MATCH IMAGES TO JSON RECORDS ---
def match_images_to_json():
    image_filenames = set(os.listdir(IMAGE_DIR))
    for month in os.listdir(JSON_BASE_DIR):
        month_path = os.path.join(JSON_BASE_DIR, month)
        if os.path.isdir(month_path):
            for date in os.listdir(month_path):
                date_path = os.path.join(month_path, date)
                if os.path.isfile(date_path) and date_path.endswith(".json"):
                    processed_data = []
                    with open(date_path, "r") as f:
                        for line in f:
                            record = json.loads(line.strip())
                            image_filename = f"{record['now']}.jpg"
                            if image_filename in image_filenames:
                                record["image_path"] = os.path.join(IMAGE_DIR, image_filename)
                                processed_data.append(record)
                    # Save filtered records to a new file
                    filtered_path = os.path.join(month_path, f"filtered_{date}")
                    with open(filtered_path, "w") as f:
                        for record in processed_data:
                            f.write(json.dumps(record) + "\n")
                    print(f"✅ Matched & saved: {filtered_path}")
    print("🎉 All JSON files matched!")

# --- STEP 4: REMOVE OLD JSON FILES ---

if __name__ == "__main__":
    copy_files_to_image_dir()
    rename_images()
    edit_json_files()
    match_images_to_json()
    print("🚀 All steps completed!")
