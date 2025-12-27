import os
import requests
import zipfile
from pathlib import Path

def download_file(url, filename):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    zip_path = data_dir / "shanghaitech.zip"
    extract_path = data_dir / "ShanghaiTech"
    
    # Official-ish Dropbox link for ShanghaiTech
    url = "https://www.dropbox.com/scl/fi/dkj5kulc9zj0rzesslck8/ShanghaiTech_Crowd_Counting_Dataset.zip?rlkey=ymbcj50ac04uvqn8p49j9af5f&dl=1"
    
    if not extract_path.exists():
        if not zip_path.exists():
            download_file(url, zip_path)
        
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Standardize folder name
        possible_dir = data_dir / "ShanghaiTech_Crowd_Counting_Dataset"
        if possible_dir.exists():
            os.rename(possible_dir, extract_path)
            
        print(f"Dataset ready at {extract_path}")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    main()
