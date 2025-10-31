#!/usr/bin/env python3
"""
Script chuẩn bị dữ liệu: copy ảnh và tạo file captions
"""

import os
import json
import shutil
import jsonlines
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class DataPreparator:
    def __init__(self, data_dir="data", output_dir="."):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.captions_dir = self.output_dir / "captions"
        
        # Create directories
        self.images_dir.mkdir(exist_ok=True)
        self.captions_dir.mkdir(exist_ok=True)
    
    def copy_images_parallel(self, source_dir, prefix="", max_workers=8):
        """Copy ảnh song song với prefix"""
        if not source_dir.exists():
            print(f"⚠️  {source_dir} not found, skipping...")
            return 0
        
        files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
        
        def copy_one(file):
            dest = self.images_dir / f"{prefix}{file.name}"
            if not dest.exists():
                shutil.copy2(file, dest)
                return 1
            return 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(copy_one, files),
                total=len(files),
                desc=f"Copying {prefix}*"
            ))
        
        return sum(results)
    
    def prepare_coco(self):
        """Chuẩn bị COCO dataset"""
        print("\n" + "="*60)
        print("📦 PREPARING COCO DATASET")
        print("="*60)
        
        coco_dir = self.data_dir / "coco"
        
        # Copy images
        print("\n📸 Copying images...")
        train_copied = self.copy_images_parallel(
            coco_dir / "train2014",
            prefix="COCO_train2014_"
        )
        val_copied = self.copy_images_parallel(
            coco_dir / "val2014",
            prefix="COCO_val2014_"
        )
        
        print(f"✅ Copied {train_copied + val_copied:,} COCO images")
        
        # Process captions
        print("\n📝 Processing captions...")
        self.process_coco_captions(coco_dir)

    
    def process_coco_captions(self, coco_dir):
        """Xử lý COCO captions"""
        annotations_dir = coco_dir / "annotations"
        
        for split in ["train", "val"]:
            caption_file = annotations_dir / f"captions_{split}2014.json"
            
            if not caption_file.exists():
                print(f"⚠️  {caption_file} not found")
                continue
            
            with open(caption_file, 'r') as f:
                data = json.load(f)
            
            # Create image_id to filename mapping
            image_dict = {img['id']: img['file_name'] for img in data['images']}
            
            # Create JSONL
            output_file = self.captions_dir / f"coco_{split}.jsonl"
            with jsonlines.open(output_file, 'w') as writer:
                for ann in tqdm(data['annotations'], desc=f"COCO {split}"):
                    image_id = ann['image_id']
                    file_name = image_dict.get(image_id)
                    if file_name:
                        # Rename with prefix
                        new_name = f"COCO_{split}2014_{str(image_id).zfill(12)}.jpg"
                        writer.write({
                            'image': new_name,
                            'caption': ann['caption']
                        })
            
            print(f"✅ Created {output_file}")
    
    def prepare_flickr(self):
        """Chuẩn bị Flickr30k dataset"""
        print("\n" + "="*60)
        print("📦 PREPARING FLICKR30K DATASET")
        print("="*60)
        
        flickr_dir = self.data_dir / "flickr"
        images_dir = flickr_dir / "flickr30k_images" / "flickr30k_images"
        
        if not images_dir.exists():
            images_dir = flickr_dir / "flickr30k_images"
        
        # Copy images
        print("\n📸 Copying images...")
        copied = self.copy_images_parallel(images_dir, prefix="flickr-")
        print(f"✅ Copied {copied:,} Flickr images")
        
        # Process captions
        print("\n📝 Processing captions...")
        captions_file = flickr_dir / "results.csv"
        
        if captions_file.exists():
            df = pd.read_csv(captions_file, delimiter='|')
            df.columns = df.columns.str.strip()
            
            output_file = self.captions_dir / "flickr.jsonl"
            with jsonlines.open(output_file, 'w') as writer:
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Flickr"):
                    writer.write({
                        'image': f"flickr-{row['image_name'].strip()}",
                        'caption': row[' comment'].strip()
                    })
            
            print(f"✅ Created {output_file}")
        else:
            print(f"⚠️  {captions_file} not found")
    
    def prepare_ktvic(self):
        """Chuẩn bị KTVIC dataset"""
        print("\n" + "="*60)
        print("📦 PREPARING KTVIC DATASET")
        print("="*60)
        
        ktvic_dir = self.data_dir / "ktvic"
        
        # Copy images
        print("\n📸 Copying images...")
        train_copied = self.copy_images_parallel(
            ktvic_dir / "train-images",
            prefix="ktvic-"
        )
        test_copied = self.copy_images_parallel(
            ktvic_dir / "public-test-images",
            prefix="ktvic-"
        )
        print(f"✅ Copied {train_copied + test_copied:,} KTVIC images")
        
        # Process captions
        print("\n📝 Processing captions...")
        for split, filename in [("train", "train_data.json"), ("test", "test_data.json")]:
            json_file = ktvic_dir / filename
            
            if not json_file.exists():
                continue
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create mapping
            image_dict = {img['id']: img['filename'] for img in data['images']}
            
            output_file = self.captions_dir / f"ktvic_{split}.jsonl"
            with jsonlines.open(output_file, 'w') as writer:
                for ann in tqdm(data['annotations'], desc=f"KTVIC {split}"):
                    image_id = ann['image_id']
                    filename = image_dict.get(image_id)
                    if filename:
                        writer.write({
                            'image': f"ktvic-{filename}",
                            'caption': ann['caption']
                        })
            
            print(f"✅ Created {output_file}")
    
    def prepare_openviic(self):
        """Chuẩn bị OpenViIC dataset"""
        print("\n" + "="*60)
        print("📦 PREPARING OPENVIIC DATASET")
        print("="*60)
        
        openviic_dir = self.data_dir / "openviic"
        
        # Copy images
        print("\n📸 Copying images...")
        copied = self.copy_images_parallel(
            openviic_dir / "images",
            prefix="openviic-"
        )
        print(f"✅ Copied {copied:,} OpenViIC images")
        
        # Process captions
        print("\n📝 Processing captions...")
        for split, filename in [
            ("train", "uit-openviic-annotation-train.json"),
            ("dev", "uit-openviic-annotation-dev.json"),
            ("test", "uit-openviic-annotation-test.json")
        ]:
            json_file = openviic_dir / filename
            
            if not json_file.exists():
                continue
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            output_file = self.captions_dir / f"openviic_{split}.jsonl"
            with jsonlines.open(output_file, 'w') as writer:
                for image_path, image_data in tqdm(data.items(), desc=f"OpenViIC {split}"):
                    for caption in image_data['captions']:
                        writer.write({
                            'image': f"openviic-{image_path}",
                            'caption': caption
                        })
            
            print(f"✅ Created {output_file}")
    
    def merge_captions(self):
        """Gộp tất cả captions thành train.jsonl và val.jsonl"""
        print("\n" + "="*60)
        print("🔗 MERGING CAPTIONS")
        print("="*60)
        
        # Train files
        train_files = [
            "coco_train.jsonl",
            "flickr.jsonl",
            "ktvic_train.jsonl",
            "openviic_train.jsonl"
        ]
        
        # Val files
        val_files = [
            "coco_val.jsonl",
            "ktvic_test.jsonl",
            "openviic_dev.jsonl"
        ]
        
        # Merge train
        train_output = self.output_dir / "train.jsonl"
        train_count = 0
        with jsonlines.open(train_output, 'w') as writer:
            for filename in train_files:
                filepath = self.captions_dir / filename
                if filepath.exists():
                    with jsonlines.open(filepath) as reader:
                        for line in reader:
                            writer.write(line)
                            train_count += 1
        
        print(f"✅ Created train.jsonl with {train_count:,} samples")
        
        # Merge val
        val_output = self.output_dir / "val.jsonl"
        val_count = 0
        with jsonlines.open(val_output, 'w') as writer:
            for filename in val_files:
                filepath = self.captions_dir / filename
                if filepath.exists():
                    with jsonlines.open(filepath) as reader:
                        for line in reader:
                            writer.write(line)
                            val_count += 1
        
        print(f"✅ Created val.jsonl with {val_count:,} samples")
    
    def prepare_all(self):
        """Chuẩn bị tất cả datasets"""
        print("\n" + "="*60)
        print("🚀 PREPARING ALL DATASETS")
        print("="*60)
        
        self.prepare_coco()
        self.prepare_flickr()
        self.prepare_ktvic()
        self.prepare_openviic()
        self.merge_captions()
        
        # Summary
        print("\n" + "="*60)
        print("✅ DATA PREPARATION COMPLETE!")
        print("="*60)
        
        image_count = len(list(self.images_dir.glob("*")))
        print(f"\n📊 Summary:")
        print(f"  Images: {image_count:,}")
        print(f"  Location: {self.images_dir.absolute()}")
        print(f"  Train captions: train.jsonl")
        print(f"  Val captions: val.jsonl")
        
        print("\n📝 Next step:")
        print("  python phoclip.py")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare data for PhoCLIP training"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing downloaded datasets"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for processed data"
    )
    
    args = parser.parse_args()
    
    preparator = DataPreparator(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    preparator.prepare_all()


if __name__ == "__main__":
    main()
