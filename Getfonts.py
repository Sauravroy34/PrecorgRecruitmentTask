import os
import shutil

# 1. Setup paths
source_dir = "/home/saurav/Desktop/PrecorgTask/fonts"
dest_dir = "available_fonts"

# 2. Create the destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

print(f"Scanning: {source_dir}")

# 3. Walk through the source directory
count = 0
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # Check if file ends with .ttf (case-insensitive)
        if file.lower().endswith(".ttf"):
            
            # Construct full file paths
            source_file_path = os.path.join(root, file)
            destination_file_path = os.path.join(dest_dir, file)
            
            try:
                # Copy the file
                shutil.copy2(source_file_path, destination_file_path)
                print(f"Copied: {file}")
                count += 1
            except Exception as e:
                print(f"Error copying {file}: {e}")

print("---")
print(f"Done! Copied {count} fonts to '{dest_dir}'")
