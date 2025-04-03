import os

def count_images_in_folders(base_folder):
    total_images = 0
    folder_counts = {}

    # Iterate through subfolders
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        
        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # Count images in the current subfolder
            image_count = sum(1 for file in os.listdir(subfolder_path) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')))
            
            folder_counts[subfolder] = image_count
            total_images += image_count

    return total_images, folder_counts

# Base folder containing the images
base_folder = "/home/sharma/projects/Sign-Language-Recognition-main/dataset/images"

# Get the counts
total_images, folder_counts = count_images_in_folders(base_folder)

# Display the results
print("Total images:", total_images)
print("Images per folder:")
for folder, count in folder_counts.items():
    print(f"  {folder}: {count}")


