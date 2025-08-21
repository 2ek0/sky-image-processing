import os
import json
from datetime import datetime

# Directory containing JSON files
base_dir = r"D:\2025\json"

# Function to process a single JSON file
def process_json_file(file_path):
    processed_data = []

    # Define cutoff datetimes
    cutoff1 = datetime(2025, 3, 14, 13, 0, 0)  # 14 Mar 2024, 13:00:00
    cutoff2 = datetime(2025, 4, 28, 10, 40, 0)  # 28 Apr 2025, 10:40:00

    with open(file_path, "r") as f:
        for line in f:
            record = json.loads(line.strip())

            # Parse the datetime
            dt = datetime.strptime(record["now"], "%Y%m%d_%H%M%S")

            # Determine irradiance factor based on datetime
            if dt < cutoff1:
                factor = 1500 / 65536
            elif dt < cutoff2:
                factor = 1200 / 65536
            else:
                factor = 1200 / 32768

            # Convert irradiance
            record["irradiance"] = round(record["irradiance"] * factor, 2)

            # Format the time as YYYYMMDD_HHMMSS
            formatted_time = dt.strftime("%Y%m%d_%H%M%S")

            # Adjust the hour by -8 (without changing the date)
            # adjusted_hour = dt.hour - 8
            # adjusted_time = f"{dt.strftime('%Y%m%d')}_{adjusted_hour:02d}{dt.strftime('%M%S')}"
            adjusted_time = f"{dt.strftime('%Y%m%d')}_{dt.hour:02d}{dt.strftime('%M%S')}"

            # Update the record
            record["now"] = adjusted_time

            # Store processed data
            processed_data.append(record)

    # Write back processed data
    if processed_data:
        with open(file_path, "w") as f:
            for record in processed_data:
                f.write(json.dumps(record) + "\n")

    print(f"✅ Processed: {file_path}")

# Loop through all JSON files in subdirectories
for month in os.listdir(base_dir):
    month_path = os.path.join(base_dir, month)
    if os.path.isdir(month_path):
        for date in os.listdir(month_path):
            date_path = os.path.join(month_path, date)
            if os.path.isfile(date_path) and date_path.endswith(".json"):
                process_json_file(date_path)

print("🎉 All JSON files processed successfully!")
