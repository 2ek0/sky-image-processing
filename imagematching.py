# import json
# import os
# from datetime import datetime

# # Base directories
# JSON_BASE_DIR = "D:\\{}\\json\\{}\\{}"  # JSON path template
# IMAGE_BASE_DIR = "D:\\{}\\{}\\{}\\{}"  # Image folder template

# # Function to convert absolute time (08:00 AM onwards) to relative HHMMSS format
# def get_relative_time(hour, minute, second):
#     relative_hour = hour - 8  # Convert absolute hour to relative (8AM -> 000000)
#     return f"{relative_hour:02d}{minute:02d}{second:02d}"

# # Function to process a single day's JSON file
# def process_json_file(year, month, day):
#     json_path = JSON_BASE_DIR.format(year, month, day)

#     if not os.path.exists(json_path):
#         print(f"JSON file not found: {json_path}")
#         return

#     with open(json_path, "r") as file:
#         data = [json.loads(line) for line in file]

#     filtered_data = []

#     for entry in data:
#         timestamp_str = entry["now"]  # Example: "2025-03-01 09:00:00.014875"
#         timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")

#         # Convert to required format
#         formatted_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")

#         # Only process entries between 08:00 and 17:59
#         if 8 <= timestamp.hour < 18:
#             relative_time = get_relative_time(timestamp.hour, timestamp.minute, timestamp.second)
#             # image_folder = IMAGE_BASE_DIR.format(year, month, day, f"{timestamp.hour:02d}")  # Correct hourly folder
#             image_folder = IMAGE_BASE_DIR.format(year, month, day, 00)  # Correct hourly folder

#             # Construct expected image filename pattern
#             image_prefix = f"201D_CAM1_{timestamp.strftime('%Y%m%d')}_{relative_time}_01"

#             # Find closest matching image
#             image_path = None
#             if os.path.exists(image_folder):
#                 images = sorted(os.listdir(image_folder))  # Sort to ensure order
#                 for file in images:
#                     if file.startswith(image_prefix) and file.endswith(".jpg"):
#                         image_path = os.path.join(image_folder, file)
#                         break  # Stop after finding the first match

#             # Add image path if found
#             if image_path:
#                 entry["image_path"] = image_path
#                 filtered_data.append(entry)

#     # Save the updated JSON file
#     output_path = f"D:\\{year}\\json\\{month}\\filtered_{day}"
#     with open(output_path, "w") as out_file:
#         json.dump(filtered_data, out_file, indent=4)

#     print(f"Processed and saved: {output_path}")

# # Process all JSON files for a given year
# def process_year(year):
#     for month in range(1, 13):
#         for day in range(1, 32):
#             month_str = f"{month:02d}"
#             day_str = f"{day:02d}.json"
#             process_json_file(year, month_str, day_str)

# # Run for 2025
# process_year(2025)

# import json
# import os
# from datetime import datetime

# # Base directories
# JSON_BASE_DIR = "D:\\{}\\json\\{}\\{}"  # JSON path template
# IMAGE_BASE_DIR = "D:\\{}\\{}\\{}\\{}"  # Image folder template

# # Function to convert absolute time (08:00 AM onwards) to relative HHMMSS format
# def get_relative_time(hour, minute, second):
#     relative_hour = hour - 8  # Convert absolute hour to relative (08:00 AM → 000000)
#     return f"{relative_hour:02d}{minute:02d}{second:02d}"

# # Function to process a single day's JSON file
# def process_json_file(year, month, day):
#     json_path = JSON_BASE_DIR.format(year, month, day)

#     if not os.path.isfile(json_path):
#         print(f"JSON file not found: {json_path}")
#         return

#     with open(json_path, "r") as file:
#         data = [json.loads(line) for line in file]

#     updated_data = []

#     for entry in data:
#         timestamp_str = entry["now"]  # Example: "2025-03-01 09:00:00.014875"
#         timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")

#         # Only process entries between 08:00 and 17:59
#         if 8 <= timestamp.hour < 18:
#             relative_time = get_relative_time(timestamp.hour, timestamp.minute, timestamp.second)
#             day = day.replace(".json", "")
#             image_folder = IMAGE_BASE_DIR.format(year, month, day, f"{timestamp.hour:02d}")  # Correct hourly folder

#             # Ensure image folder exists
#             if not os.path.exists(image_folder):
#                 print(f"Image folder not found: {image_folder}")
#                 updated_data.append(entry)  # Keep the entry as is
#                 continue  # Skip to next entry

#             # Construct expected image filename pattern
#             expected_filename = f"201D_CAM1_{timestamp.strftime('%Y%m%d')}_{relative_time}_01"

#             # Find the closest matching image file
#             image_path = None
#             images = sorted(os.listdir(image_folder))  # Sort to ensure order
#             for file in images:
#                 if file.startswith(expected_filename) and file.endswith(".jpg"):
#                     image_path = os.path.join(image_folder, file)
#                     break  # Stop after finding the first match

#             # If a matching image is found, add "image_path" to the entry
#             if image_path:
#                 entry["image_path"] = image_path

#         updated_data.append(entry)  # Append updated entry

#     # Save the updated JSON file with the new "image_path"
#     output_path = f"D:\\{year}\\json\\{month}\\filtered_{day}"
#     with open(output_path, "w") as out_file:
#         json.dump(updated_data, out_file, indent=4)

#     print(f"Processed and saved: {output_path}")

# # Process all JSON files for a given year
# def process_year(year):
#     for month in range(1, 13):
#         for day in range(1, 32):
#             month_str = f"{month:02d}"
#             day_str = f"{day:02d}.json"
#             process_json_file(year, month_str, day_str)

# # Run for 2025
# process_year(2025)

import os
import json

# Directories
json_base_dir = r"D:\2025\json"
image_dir = r"D:\Image"

# Get all image filenames as a set for fast lookup
image_filenames = set(os.listdir(image_dir))

# Function to process a single JSON file
def process_json_file(file_path):
    processed_data = []

    with open(file_path, "r") as f:
        for line in f:
            record = json.loads(line.strip())

            # Check if an image file exists matching the "now" value
            image_filename = f"{record['now']}.jpg"
            if image_filename in image_filenames:
                record["image_path"] = os.path.join(image_dir, image_filename)
                processed_data.append(record)

    # Overwrite the JSON file with only the matched records
    if processed_data:
        with open(file_path, "w") as f:
            for record in processed_data:
                f.write(json.dumps(record) + "\n")

    print(f"✅ Processed: {file_path}")

# Loop through all JSON files in subdirectories
for month in os.listdir(json_base_dir):
    month_path = os.path.join(json_base_dir, month)
    if os.path.isdir(month_path):
        for day in os.listdir(month_path):
            day_path = os.path.join(month_path, day)
            if os.path.isfile(day_path) and day_path.endswith(".json"):
                process_json_file(day_path)

print("🎉 All JSON files processed successfully!")
