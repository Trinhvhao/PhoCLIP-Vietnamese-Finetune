#!/usr/bin/env python3
"""
Script tự động tải và chuẩn bị dữ liệu cho PhoCLIP
Hỗ trợ: COCO, Flickr30k, KTVIC, OpenViIC
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import requests
import zipfile
import shutil

class DataCrawler:
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def download_file(self, url, dest_path, desc="Downloading"):
        """Tải file với progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    
    def extract_zip(self, zip_path, extract_to):
        """Giải nén file zip"""
        print(f"📦 Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Extraction complete")
    
    def download_coco(self):
        """Tải COCO dataset"""
        print("\n" + "="*60)
        print("📥 DOWNLOADING COCO DATASET")
        print("="*60)
        
        coco_dir = self.base_dir / "coco"
        coco_dir.mkdir(exist_ok=True)
        
        # URLs
        urls = {
            "train2014": "http://images.cocodataset.org/zips/train2014.zip",
            "val2014": "http://images.cocodataset.org/zips/val2014.zip",
            "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
        }
        
        for name, url in urls.items():
            zip_path = coco_dir / f"{name}.zip"
            
            if zip_path.exists():
                print(f"⏭️  {name}.zip already exists, skipping download")
            else:
                print(f"\n📥 Downloading {name}...")
                self.download_file(url, zip_path, desc=name)
            
            # Extract
            if not (coco_dir / name).exists():
                self.extract_zip(zip_path, coco_dir)
                # Clean up zip
                zip_path.unlink()
        
        print("\n✅ COCO dataset downloaded successfully!")
        print(f"📂 Location: {coco_dir.absolute()}")

    
    def download_flickr(self):
        """Hướng dẫn tải Flickr30k (cần Kaggle)"""
        print("\n" + "="*60)
        print("📥 FLICKR30K DATASET")
        print("="*60)
        
        flickr_dir = self.base_dir / "flickr"
        flickr_dir.mkdir(exist_ok=True)
        
        print("\n⚠️  Flickr30k requires Kaggle account")
        print("\n📝 Steps to download:")
        print("1. Go to: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset")
        print("2. Click 'Download' button")
        print("3. Extract to:", flickr_dir.absolute())
        print("\nOr use Kaggle API:")
        print(f"  kaggle datasets download -d hsankesara/flickr-image-dataset -p {flickr_dir}")
        print(f"  unzip {flickr_dir}/flickr-image-dataset.zip -d {flickr_dir}")
        
        # Check if already downloaded
        if (flickr_dir / "flickr30k_images").exists():
            print("\n✅ Flickr30k images found!")
        else:
            print("\n❌ Flickr30k images not found. Please download manually.")
    
    def download_ktvic(self):
        """Tải KTVIC dataset"""
        print("\n" + "="*60)
        print("📥 DOWNLOADING KTVIC DATASET")
        print("="*60)
        
        ktvic_dir = self.base_dir / "ktvic"
        
        if ktvic_dir.exists():
            print("⏭️  KTVIC already exists, skipping...")
            return
        
        print("\n📥 Cloning KTVIC repository...")
        try:
            subprocess.run([
                "git", "clone",
                "https://github.com/uitnlp/KTVIC.git",
                str(ktvic_dir)
            ], check=True)
            print("✅ KTVIC downloaded successfully!")
            print(f"📂 Location: {ktvic_dir.absolute()}")
        except subprocess.CalledProcessError:
            print("❌ Failed to clone KTVIC. Please install git or download manually.")
            print("   URL: https://github.com/uitnlp/KTVIC")
    
    def download_openviic(self):
        """Tải OpenViIC dataset"""
        print("\n" + "="*60)
        print("📥 DOWNLOADING OPENVIIC DATASET")
        print("="*60)
        
        openviic_dir = self.base_dir / "openviic"
        
        if openviic_dir.exists():
            print("⏭️  OpenViIC already exists, skipping...")
            return
        
        print("\n📥 Cloning OpenViIC repository...")
        try:
            subprocess.run([
                "git", "clone",
                "https://github.com/uitnlp/OpenViIC.git",
                str(openviic_dir)
            ], check=True)
            print("✅ OpenViIC downloaded successfully!")
            print(f"📂 Location: {openviic_dir.absolute()}")
        except subprocess.CalledProcessError:
            print("❌ Failed to clone OpenViIC. Please install git or download manually.")
            print("   URL: https://github.com/uitnlp/OpenViIC")
    
    def verify_data(self):
        """Kiểm tra tính toàn vẹn của dữ liệu"""
        print("\n" + "="*60)
        print("🔍 VERIFYING DATA")
        print("="*60)
        
        datasets = {
            "COCO train2014": self.base_dir / "coco" / "train2014",
            "COCO val2014": self.base_dir / "coco" / "val2014",
            "Flickr30k": self.base_dir / "flickr" / "flickr30k_images",
            "KTVIC": self.base_dir / "ktvic" / "train-images",
            "OpenViIC": self.base_dir / "openviic" / "images"
        }
        
        results = []
        for name, path in datasets.items():
            if path.exists():
                count = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.png")))
                results.append((name, "✅", count))
            else:
                results.append((name, "❌", 0))
        
        print("\n📊 Dataset Status:")
        print("-" * 60)
        for name, status, count in results:
            print(f"{status} {name:20s} - {count:,} images")
        print("-" * 60)
        
        total = sum(r[2] for r in results)
        print(f"\n📈 Total images: {total:,}")
    
    def download_all(self):
        """Tải tất cả datasets"""
        print("\n" + "="*60)
        print("🚀 DOWNLOADING ALL DATASETS")
        print("="*60)
        
        self.download_coco()
        self.download_flickr()
        self.download_ktvic()
        self.download_openviic()
        self.verify_data()
        
        print("\n" + "="*60)
        print("✅ ALL DOWNLOADS COMPLETE!")
        print("="*60)
        print("\n📝 Next steps:")
        print("1. Run: python prepare_data.py")
        print("2. Run: python phoclip.py")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for PhoCLIP training"
    )
    parser.add_argument(
        "--dataset",
        choices=["coco", "flickr", "ktvic", "openviic"],
        help="Download specific dataset"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded data"
    )
    parser.add_argument(
        "--base-dir",
        default="data",
        help="Base directory for datasets (default: data)"
    )
    
    args = parser.parse_args()
    
    crawler = DataCrawler(base_dir=args.base_dir)
    
    if args.verify:
        crawler.verify_data()
    elif args.all:
        crawler.download_all()
    elif args.dataset:
        if args.dataset == "coco":
            crawler.download_coco()
        elif args.dataset == "flickr":
            crawler.download_flickr()
        elif args.dataset == "ktvic":
            crawler.download_ktvic()
        elif args.dataset == "openviic":
            crawler.download_openviic()
    else:
        parser.print_help()
        print("\n💡 Examples:")
        print("  python crawl_data.py --all")
        print("  python crawl_data.py --dataset coco")
        print("  python crawl_data.py --verify")


if __name__ == "__main__":
    main()
