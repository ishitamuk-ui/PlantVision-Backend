import os
import urllib.request
import zipfile
import shutil
import time

def download_and_extract_dataset():
    """
    Downloads the PlantVillage dataset from the popular community-maintained 
    GitHub repository and extracts the color images into data/PlantVillage.
    """
    # The URL to the zip file of the repository
    url = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
    zip_path = "plantvillage_master.zip"
    extract_dir = "PlantVillage-Dataset-master"
    target_dir = os.path.join("data", "PlantVillage")
    source_images_dir = os.path.join(extract_dir, "raw", "color")
    
    # Make sure 'data' directory exists
    os.makedirs("data", exist_ok=True)
    
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"Dataset already appears to exist at {target_dir}")
        return

    print("Downloading PlantVillage dataset (this may take a few minutes)...")
    
    # Download with progress representation
    def reporthook(count, block_size, total_size):
        if count == 0:
            start_time = time.time()
            return
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading: {percent}%", end="")
        
    urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
    print("\nDownload complete.")
    
    print("Extracting ZIP file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Extraction complete.")
    
    print(f"Moving images to {target_dir}...")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    # Move the 'color' directory to be our 'data/PlantVillage' directory
    shutil.move(source_images_dir, target_dir)
    
    print("Cleaning up temporary files...")
    os.remove(zip_path)
    shutil.rmtree(extract_dir)
    
    num_classes = len(os.listdir(target_dir))
    print(f"\nSuccessfully set up PlantVillage dataset at '{target_dir}' with {num_classes} classes!")
    print("You are now ready to run 'python scripts/train_all.py'")

if __name__ == "__main__":
    download_and_extract_dataset()
