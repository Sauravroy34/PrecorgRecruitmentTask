import shutil
import os

# Configuration
FOLDER_TO_ZIP = "dataset_task2"  # The folder you want to zip
OUTPUT_FILENAME = "dataset_task2" # The name of the zip file (without .zip extension)

# Check if folder exists
if os.path.exists(FOLDER_TO_ZIP):
    print(f"Zipping '{FOLDER_TO_ZIP}'...")
    
    # create the zip file
    shutil.make_archive(OUTPUT_FILENAME, 'zip', FOLDER_TO_ZIP)
    
    print(f"Success! Created '{OUTPUT_FILENAME}.zip'")
else:
    print(f"Error: Folder '{FOLDER_TO_ZIP}' not found.")
