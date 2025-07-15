import requests
import zipfile
import os

import os
import zipfile
import requests

def download_and_extract(zip_url, output_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)

    target_pth = os.path.join(extract_to, os.path.basename(output_path).replace(".zip", ".pth"))
    if not os.path.exists(target_pth):
        print(f"Downloading model zip to: {output_path}")
        response = requests.get(zip_url)
        with open(output_path, "wb") as f:
            f.write(response.content)

        print("Extracting...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to {extract_to}")

        os.remove(output_path)
        print(f"Deleted {output_path}")
    else:
        print(f"{target_pth} already exists.")


def download_all_models():
    extract_path = "CCA/models/trained_models"

    download_and_extract(
        zip_url="https://github.com/dxn1405/Cervical-Cancer-Image-Classification/releases/download/v1.0/Swin_best.zip",
        output_path=f"{extract_path}/Swin_best.zip",
        extract_to=extract_path
    )

    download_and_extract(
        zip_url="https://github.com/dxn1405/Cervical-Cancer-Image-Classification/releases/download/v1.0/DeiT_best.zip",
        output_path=f"{extract_path}/DeiT_best.zip",
        extract_to=extract_path
    )

    download_and_extract(
        zip_url="https://github.com/dxn1405/Cervical-Cancer-Image-Classification/releases/download/v1.0/CrossViT_best.zip",
        output_path=f"{extract_path}/CrossViT_best.zip",
        extract_to=extract_path
    )

if __name__ == "__main__":
    download_all_models()
