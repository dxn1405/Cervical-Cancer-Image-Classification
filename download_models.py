import gdown
import zipfile
import os

def download_and_extract(zip_url, output_path, extract_to):
    target_pth = os.path.join(extract_to, os.path.basename(output_path).replace(".zip", ".pth"))
    if not os.path.exists(target_pth):
        print(f"Downloading model zip to: {output_path}")
        gdown.download(zip_url, output_path, quiet=False, fuzzy=True)

        print("Extracting...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to {extract_to}")

        os.remove(output_path)
        print(f"Deleted {output_path}")
    else:
        print(f"{target_pth} already exists.")


def download_all_models():
    # Always resolve path relative to this file (download_models.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    extract_path = os.path.join(base_dir, "CCA", "models", "trained_models")

    os.makedirs(extract_path, exist_ok=True)


    # ✅ Model 1: Swin
    download_and_extract(
        zip_url="https://drive.google.com/uc?id=1IZRMb6_EQJ6RbEZiYpNaI3NHWSI0VCLR",
        output_path=f"{extract_path}/Swin_best.zip",
        extract_to=extract_path
    )

    # ✅ Model 2: DeiT
    download_and_extract(
        zip_url="https://drive.google.com/uc?id=1IZRMb6_EQJ6RbEZiYpNaI3NHWSI0VCLR",
        output_path=f"{extract_path}/DeiT_best.zip",
        extract_to=extract_path
    )

    # ✅ Model 3: CrossViT
    download_and_extract(
        zip_url="https://drive.google.com/uc?id=1IZRMb6_EQJ6RbEZiYpNaI3NHWSI0VCLR",
        output_path=f"{extract_path}/CrossViT_best.zip",
        extract_to=extract_path
    )
