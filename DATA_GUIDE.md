# 📦 Hướng Dẫn Tải và Chuẩn Bị Dữ Liệu

## 📋 Tổng Quan Datasets

PhoCLIP được huấn luyện trên 4 datasets chính:

| Dataset | Số Ảnh | Số Captions | Nguồn | Ngôn Ngữ |
|---------|--------|-------------|-------|----------|
| **COCO** | 123,287 | 616,767 | MS COCO (dịch sang tiếng Việt) | 🇻🇳 |
| **Flickr30k** | 31,783 | 158,915 | Flickr30k (dịch sang tiếng Việt) | 🇻🇳 |
| **KTVIC** | 4,327 | 21,635 | UIT-KTVIC | 🇻🇳 |
| **OpenViIC** | 9,328 | 46,640 | UIT-OpenViIC | 🇻🇳 |
| **TỔNG** | **168,725** | **843,957** | - | 🇻🇳 |

---

## 🚀 Cài Đặt Công Cụ Tải Dữ Liệu

```bash
# Cài đặt gdown để tải từ Google Drive
pip install gdown

# Cài đặt wget (nếu chưa có)
pip install wget

# Cài đặt các thư viện cần thiết
pip install jsonlines pandas tqdm
```

---

## 📥 1. COCO Dataset (Tiếng Việt)

### Tải Ảnh COCO

```bash
# Tạo thư mục
mkdir -p data/coco

# Tải train2014 (13GB)
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip -d data/coco/
rm train2014.zip

# Tải val2014 (6GB)
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d data/coco/
rm val2014.zip
```

### Tải Captions Tiếng Việt

```bash
# Tải captions đã dịch sang tiếng Việt
# (Bạn cần có file này từ nguồn dịch thuật)
# Hoặc sử dụng script crawl_data.py để tự động tải
python crawl_data.py --dataset coco
```

**Cấu trúc thư mục:**
```
data/coco/
├── train2014/
│   ├── COCO_train2014_000000000009.jpg
│   ├── COCO_train2014_000000000025.jpg
│   └── ...
├── val2014/
│   ├── COCO_val2014_000000000042.jpg
│   └── ...
├── captions_train2014.json
└── captions_val2014.json
```

---

## 📥 2. Flickr30k Dataset (Tiếng Việt)

### Tải Ảnh Flickr30k

```bash
# Tạo thư mục
mkdir -p data/flickr

# Tải ảnh Flickr30k (yêu cầu đăng ký tại Kaggle)
# Link: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

# Sau khi tải về, giải nén:
unzip flickr30k-images.zip -d data/flickr/
```

### Tải Captions Tiếng Việt

```bash
# Sử dụng script tự động
python crawl_data.py --dataset flickr
```

**Cấu trúc thư mục:**
```
data/flickr/
├── Images/
│   ├── 1000092795.jpg
│   ├── 10002456.jpg
│   └── ...
└── captions_vi.txt
```

**Format file captions_vi.txt:**
```
1000092795.jpg	Một người đàn ông đang chơi guitar trên sân khấu
1000092795.jpg	Ca sĩ biểu diễn với cây đàn guitar điện
...
```

---

## 📥 3. KTVIC Dataset

### Tải KTVIC

```bash
# Tạo thư mục
mkdir -p data/ktvic

# Clone repository
git clone https://github.com/uitnlp/KTVIC.git data/ktvic/

# Hoặc tải trực tiếp
python crawl_data.py --dataset ktvic
```

**Cấu trúc thư mục:**
```
data/ktvic/
├── train-images/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── public-test-images/
│   └── ...
├── train_data.json
└── test_data.json
```

**Format JSON:**
```json
{
  "images": [
    {"id": 1, "filename": "1.jpg"}
  ],
  "annotations": [
    {"image_id": 1, "caption": "Một chiếc xe hơi màu đỏ"}
  ]
}
```

---

## 📥 4. OpenViIC Dataset

### Tải OpenViIC

```bash
# Tạo thư mục
mkdir -p data/openviic

# Clone repository
git clone https://github.com/uitnlp/OpenViIC.git data/openviic/

# Hoặc tải trực tiếp
python crawl_data.py --dataset openviic
```

**Cấu trúc thư mục:**
```
data/openviic/
├── images/
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ...
├── uit-openviic-annotation-train.json
├── uit-openviic-annotation-dev.json
└── uit-openviic-annotation-test.json
```

**Format JSON:**
```json
{
  "000000000001.jpg": {
    "captions": [
      "Một con mèo đang ngủ trên ghế sofa",
      "Con mèo nằm nghỉ ngơi trong phòng khách"
    ]
  }
}
```

---

## 🔧 Sử Dụng Script Tự Động

### Tải Tất Cả Datasets

```bash
# Tải tất cả datasets cùng lúc
python crawl_data.py --all

# Hoặc tải từng dataset riêng
python crawl_data.py --dataset coco
python crawl_data.py --dataset flickr
python crawl_data.py --dataset ktvic
python crawl_data.py --dataset openviic
```

### Kiểm Tra Dữ Liệu

```bash
# Kiểm tra tính toàn vẹn của dữ liệu
python crawl_data.py --verify
```

---

## 📊 Chuẩn Bị Dữ Liệu Cho Training

Sau khi tải xong, chạy script để chuẩn bị dữ liệu:

```bash
# Tổng hợp tất cả datasets vào thư mục images/
python prepare_data.py

# Tạo file JSONL cho training
python create_jsonl.py
```

**Kết quả:**
```
images/
├── COCO_train2014_000000000009.jpg
├── COCO_val2014_000000000042.jpg
├── flickr-1000092795.jpg
├── ktvic-1.jpg
├── openviic-000000000001.jpg
└── ...

captions/
├── train.jsonl
└── val.jsonl
```

**Format JSONL:**
```json
{"image": "COCO_train2014_000000000009.jpg", "caption": "Một chiếc xe hơi màu đỏ đậu trước ngôi nhà"}
{"image": "flickr-1000092795.jpg", "caption": "Người đàn ông chơi guitar trên sân khấu"}
```

---

## ⚠️ Lưu Ý Quan Trọng

### Dung Lượng Ổ Cứng

- **COCO**: ~19GB (ảnh) + ~500MB (captions)
- **Flickr30k**: ~5GB (ảnh) + ~50MB (captions)
- **KTVIC**: ~500MB
- **OpenViIC**: ~1GB
- **Tổng cộng**: ~26GB

### Quyền Truy Cập

- **COCO**: Public, tải trực tiếp
- **Flickr30k**: Cần đăng ký Kaggle
- **KTVIC**: Public GitHub
- **OpenViIC**: Public GitHub

### Thời Gian Tải

- Tốc độ mạng 100Mbps: ~30-45 phút
- Tốc độ mạng 50Mbps: ~1-1.5 giờ
- Tốc độ mạng 10Mbps: ~5-6 giờ

---

## 🐛 Xử Lý Lỗi Thường Gặp

### Lỗi: "Connection timeout"

```bash
# Tăng timeout
export WGET_TIMEOUT=300
wget --timeout=300 [URL]
```

### Lỗi: "Disk space full"

```bash
# Kiểm tra dung lượng
df -h

# Xóa file tạm
rm -rf /tmp/*
```

### Lỗi: "Corrupted zip file"

```bash
# Tải lại file
rm [file].zip
wget [URL]
```

---

## 📞 Hỗ Trợ

Nếu gặp vấn đề khi tải dữ liệu:

1. Kiểm tra kết nối internet
2. Đảm bảo đủ dung lượng ổ cứng
3. Xem log chi tiết: `python crawl_data.py --verbose`
4. Mở issue trên GitHub

---

## 📚 Tài Liệu Tham Khảo

- [MS COCO Dataset](https://cocodataset.org/)
- [Flickr30k Dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
- [UIT-KTVIC](https://github.com/uitnlp/KTVIC)
- [UIT-OpenViIC](https://github.com/uitnlp/OpenViIC)
