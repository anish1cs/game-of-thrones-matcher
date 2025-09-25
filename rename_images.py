import os
import re

def rename_character_images(directory_path):
    """
    Scans a directory and renames image files to the standard format:
    'all_lowercase_with_underscores.jpg'.
    """
    print(f"--- Starting Image Renaming ---")
    print(f"Scanning directory: '{directory_path}'\n")

    # Check if the directory exists to avoid errors
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found. Please make sure a folder named '{directory_path}' exists.")
        return

    files_renamed_count = 0
    for filename in os.listdir(directory_path):
        # We process any file that is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            # Separate the name from the extension (e.g., 'Daenerys Targaryen' from '.png')
            character_name = os.path.splitext(filename)[0]

            # Apply the exact same cleaning logic as model.py
            # Replace one or more spaces/dashes with a single underscore
            clean_name = re.sub(r'[\s-]+', '_', character_name).lower()
            
            # Standardize the final extension to .jpg
            new_filename = f"{clean_name}.jpg"

            # Get the full old and new file paths
            old_filepath = os.path.join(directory_path, filename)
            new_filepath = os.path.join(directory_path, new_filename)

            # Only rename if the new name is different
            if old_filepath != new_filepath:
                try:
                    os.rename(old_filepath, new_filepath)
                    print(f"✅ Renamed: '{filename}'  ->  '{new_filename}'")
                    files_renamed_count += 1
                except OSError as e:
                    print(f"❌ Error renaming '{filename}': {e}")
            else:
                print(f"🆗 Skipped: '{filename}' is already in the correct format.")


    if files_renamed_count == 0:
        print("\nNo files needed renaming.")
    else:
        print(f"\nFinished. Successfully renamed {files_renamed_count} files.")
    print("--- Script Complete ---")


if __name__ == '__main__':
    # *** THIS IS THE CORRECTED PART ***
    # The path now correctly points to the nested folder
    IMAGES_TO_RENAME_FOLDER = os.path.join('static', 'images', 'images_to_rename')
    rename_character_images(IMAGES_TO_RENAME_FOLDER)

