import os
import urllib.request
import zipfile
import shutil

DATASET_URL = "https://zenodo.org/api/records/1188976/files/Audio_Speech_Actors_01-24.zip/content"
ZIP_NAME = "Audio_Speech_Actors_01-24.zip"

def download_and_extract(data_dir="data"):
    """
    Downloads and extracts the RAVDESS Speech audio dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, ZIP_NAME)
    
    # Check if files are already extracted
    # RAVDESS speech has Actor_01 to Actor_24 directories
    extraction_complete = True
    for i in range(1, 25):
        actor_dir = os.path.join(data_dir, f"Actor_{i:02d}")
        if not os.path.isdir(actor_dir):
            extraction_complete = False
            break
            
    if extraction_complete:
        print("[INFO] RAVDESS Speech dataset already downloaded and extracted.")
        return True

    # Check if zip exists
    if not os.path.exists(zip_path):
        print(f"[INFO] Downloading RAVDESS Speech dataset from:\n{DATASET_URL}")
        
        # Download block-by-block with progress indicator
        import ssl
        try:
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(DATASET_URL, context=context) as response, open(zip_path, 'wb') as out_file:
                total_size = int(response.info().get('Content-Length', 0))
                downloaded = 0
                block_size = 1024 * 1024  # 1MB blocks
                
                print("[INFO] Download progress:")
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloaded {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({percent:.1f}%)", end="")
                    else:
                        print(f"\rDownloaded {downloaded / (1024*1024):.1f}MB", end="")
                print("\n[INFO] Download finished successfully.")
        except Exception as e:
            print(f"\n[ERROR] Failed to download dataset: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return False
    else:
        print("[INFO] ZIP archive already exists. Skipping download.")

    # Extract zip archive
    print(f"[INFO] Extracting {zip_path} to {data_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("[INFO] Extraction completed successfully.")
        
        # Optionally remove zip file to save disk space
        print("[INFO] Removing temporary zip file to save disk space...")
        os.remove(zip_path)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to extract dataset: {e}")
        return False

if __name__ == "__main__":
    download_and_extract()
