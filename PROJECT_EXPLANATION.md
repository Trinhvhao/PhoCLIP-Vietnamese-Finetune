# 🎯 Giải Thích Chi Tiết Dự Án PhoCLIP

## 📖 Giới Thiệu Bài Toán

### Image-Text Matching (Ghép Cặp Ảnh-Văn Bản)

**Mục tiêu**: Xây dựng mô hình có khả năng:
- Tìm ảnh phù hợp từ mô tả văn bản (Text → Image)
- Tìm mô tả văn bản phù hợp từ ảnh (Image → Text)
- Đo độ tương đồng giữa ảnh và văn bản

**Ví dụ:**
```
Input: "Một chiếc xe hơi màu đỏ đậu trước ngôi nhà"
Output: [Ảnh xe hơi đỏ trước nhà]
```

### Thách Thức
- Ảnh và văn bản ở 2 không gian khác nhau
- Ít mô hình hỗ trợ tốt tiếng Việt
- Cần xử lý hàng trăm nghìn cặp ảnh-văn bản

---

## 💡 Ý Tưởng PhoCLIP

### Giải Pháp
Kết hợp:
- **PhoBERT**: Hiểu tiếng Việt
- **ResNet50**: Trích xuất đặc trưng ảnh
- **Contrastive Learning**: Học không gian chung

### Ứng Dụng
- Tìm kiếm ảnh bằng ngôn ngữ tự nhiên
- Chatbot mô tả ảnh
- Hỗ trợ người khiếm thị
- E-commerce: Tìm sản phẩm bằng mô tả

---

## 🏗️ Kiến Trúc Mô Hình


### Luồng Dữ Liệu

```
Image (224x224) ──→ ResNet50 ──→ [2048-dim] ──→ Projection ──→ [512-dim]
                                                      ↓
                                              Cosine Similarity
                                                      ↓
Text (Vietnamese) ──→ PhoBERT ──→ [1024-dim] ──→ Projection ──→ [512-dim]
```

### Các Thành Phần

**1. Image Encoder (ResNet50)**
- Input: Ảnh RGB 224x224
- Output: Vector 2048 chiều
- Pretrained trên ImageNet

**2. Text Encoder (PhoBERT-large)**
- Input: Câu tiếng Việt (max 70 tokens)
- Output: Vector 1024 chiều
- Pretrained trên corpus tiếng Việt

**3. Projection Heads**
- Chiếu cả 2 encoders về không gian 512 chiều
- Sử dụng GELU activation + LayerNorm

**4. Contrastive Loss**
- Tối đa hóa độ tương đồng của cặp (ảnh, caption) đúng
- Tối thiểu hóa độ tương đồng của các cặp sai

---

## 🔄 Luồng Code Chi Tiết

### 1. Chuẩn Bị Dữ Liệu


**File: phoclip.py (dòng 1-100)**

```python
# Bước 1: Load VnCoreNLP để phân đoạn từ tiếng Việt
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])

# Bước 2: Định nghĩa config
class CFG:
    batch_size = 64
    image_path = "images/"
    text_encoder_model = "vinai/phobert-large"
    image_encoder_model = "resnet50"
```

**Tại sao cần phân đoạn từ?**
- Tiếng Việt: "làm việc" → "làm_việc" (1 token)
- PhoBERT yêu cầu input đã được phân đoạn

### 2. Tải và Xử Lý Datasets

**File: phoclip.py (dòng 100-300)**

```python
# Tải COCO
df_coco = load_jsonl("cocopath_train.jsonl")

# Tải Flickr
df_flickr = pd.DataFrame({'image': [...], 'caption': [...]})

# Gộp tất cả
data_df = pd.concat([df_coco, df_flickr, df_ktvic, df_openviic])
```

**Xử lý ảnh:**
- Copy ảnh từ nhiều nguồn vào 1 folder `images/`
- Đổi tên với prefix: `COCO_`, `flickr-`, `ktvic-`, `openviic-`

### 3. Dataset và DataLoader

**File: phoclip.py (dòng 300-400)**


```python
class CLIPDataset:
    def __getitem__(self, idx):
        # 1. Đọc ảnh
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Transform ảnh (resize, normalize)
        image = self.transforms(image=image)['image']
        
        # 3. Phân đoạn từ tiếng Việt
        caption = CFG.segmenter(caption)  # "làm việc" → "làm_việc"
        
        # 4. Tokenize text
        encoded = tokenizer(caption, max_length=70)
        
        return {'image': image, 'input_ids': ..., 'attention_mask': ...}
```

**Augmentation:**
- Resize về 224x224
- Normalize theo ImageNet mean/std

### 4. Định Nghĩa Mô Hình

**File: phoclip.py (dòng 400-500)**

```python
class CLIPModel(nn.Module):
    def __init__(self):
        self.image_encoder = ImageEncoder()      # ResNet50
        self.text_encoder = TextEncoder()        # PhoBERT
        self.image_projection = ProjectionHead() # 2048→512
        self.text_projection = ProjectionHead()  # 1024→512
    
    def forward(self, batch):
        # Encode
        img_features = self.image_encoder(batch["image"])
        txt_features = self.text_encoder(batch["input_ids"])
        
        # Project
        img_emb = self.image_projection(img_features)
        txt_emb = self.text_projection(txt_features)
        
        # Compute loss
        return contrastive_loss(img_emb, txt_emb)
```

### 5. Contrastive Loss


**Cách hoạt động:**

```python
# Batch size = 4
images = [img1, img2, img3, img4]
texts = [txt1, txt2, txt3, txt4]

# Tính similarity matrix (4x4)
similarity = text_emb @ image_emb.T

# Targets: diagonal = 1, others = 0
# txt1 ↔ img1 = 1.0 (đúng)
# txt1 ↔ img2 = 0.0 (sai)
# txt1 ↔ img3 = 0.0 (sai)
# txt1 ↔ img4 = 0.0 (sai)

# Loss: Cross-entropy
loss = -log(softmax(similarity))
```

**Mục tiêu:**
- Cặp đúng (txt1, img1): similarity cao
- Cặp sai (txt1, img2): similarity thấp

### 6. Training Loop

**File: phoclip.py (dòng 500-600)**

```python
def train_epoch(model, train_loader, optimizer):
    for batch in train_loader:
        # Forward
        loss = model(batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Optimizer:**
- AdamW với learning rates khác nhau:
  - Image encoder: 1e-4
  - Text encoder: 1e-5
  - Projection heads: 1e-3

### 7. Inference

**File: phoclip.py (dòng 600-700)**


```python
def find_matches(model, query_text, database_embeddings, n=25):
    # 1. Encode query text
    text_emb = model.text_encoder(query_text)
    text_emb = model.text_projection(text_emb)
    
    # 2. Normalize embeddings
    text_emb = F.normalize(text_emb, p=2, dim=-1)
    db_emb = F.normalize(database_embeddings, p=2, dim=-1)
    
    # 3. Compute similarity
    similarity = text_emb @ db_emb.T
    
    # 4. Get top-k
    values, indices = torch.topk(similarity, n)
    
    return [image_filenames[idx] for idx in indices]
```

**Ví dụ sử dụng:**
```python
# Tìm ảnh từ text
matches = find_matches(
    model, 
    text="xe hơi đậu trước ngôi nhà",
    database_embeddings=all_image_embeddings,
    n=25
)
```

---

## 📊 Cách Hoạt Động (Step by Step)

### Training Phase

**Bước 1: Load Data**
```
COCO: 123,287 ảnh + 616,767 captions
Flickr: 31,783 ảnh + 158,915 captions
→ Total: 168,725 ảnh + 843,957 captions
```

**Bước 2: Preprocessing**
```
Image: Resize(224,224) → Normalize
Text: Phân đoạn từ → Tokenize (max_len=70)
```

**Bước 3: Forward Pass**
```
Batch (64 samples):
  Images [64, 3, 224, 224] → ResNet50 → [64, 2048] → Proj → [64, 512]
  Texts [64, 70] → PhoBERT → [64, 1024] → Proj → [64, 512]
```

**Bước 4: Compute Loss**
```
Similarity Matrix [64, 64]
Target: Identity matrix (diagonal = 1)
Loss: Cross-entropy
```

**Bước 5: Backward & Update**
```
loss.backward()
optimizer.step()
```

### Inference Phase


**Bước 1: Extract All Image Embeddings**
```python
# Chạy 1 lần duy nhất
for batch in dataloader:
    img_emb = model.image_encoder(batch["image"])
    img_emb = model.image_projection(img_emb)
    all_embeddings.append(img_emb)

# Lưu vào file
torch.save(all_embeddings, "image_embeddings.pt")
```

**Bước 2: Search**
```python
# Query
query = "xe hơi màu đỏ"
query_emb = encode_text(query)  # [1, 512]

# Compute similarity với tất cả ảnh
similarity = query_emb @ all_embeddings.T  # [1, 168725]

# Top-25
top_indices = similarity.topk(25)
results = [images[idx] for idx in top_indices]
```

**Tốc độ:**
- Extract embeddings: ~10 phút (1 lần duy nhất)
- Search: <0.1 giây (real-time)

---

## 🎓 Kiến Thức Cần Thiết

### 1. Deep Learning Cơ Bản
- Neural Networks
- Backpropagation
- Optimization (SGD, Adam)

### 2. Computer Vision
- CNN (Convolutional Neural Networks)
- ResNet architecture
- Image preprocessing

### 3. NLP
- Transformer architecture
- BERT model
- Tokenization

### 4. PyTorch
- nn.Module
- DataLoader
- Autograd

---

## 🚀 Ứng Dụng Thực Tế

### 1. Tìm Kiếm Ảnh Google-style
```python
query = "con mèo đang ngủ"
results = search_images(query)
# → Trả về 100 ảnh mèo ngủ
```

### 2. E-commerce
```python
query = "áo sơ mi trắng tay dài"
products = search_products(query)
# → Hiển thị sản phẩm phù hợp
```

### 3. Hỗ Trợ Người Khiếm Thị
```python
image = capture_camera()
description = describe_image(image)
text_to_speech(description)
# → "Một chiếc xe hơi đang đậu bên đường"
```

### 4. Tổ Chức Thư Viện Ảnh
```python
# Tự động gắn tag
for image in photo_library:
    tags = generate_tags(image)
    # → ["xe hơi", "đường phố", "ban ngày"]
```

---

## 📈 Kết Quả Mong Đợi

### Metrics
- **Top-1 Accuracy**: ~40-50%
- **Top-5 Accuracy**: ~70-80%
- **Top-10 Accuracy**: ~85-90%

### So Sánh
| Model | Top-5 Accuracy |
|-------|----------------|
| Random | 0.003% |
| PhoCLIP | ~75% |
| CLIP (English) | ~85% |

---

## 🔍 Debug và Tối Ưu

### Kiểm Tra Model
```python
# Test forward pass
batch = next(iter(train_loader))
loss = model(batch)
print(f"Loss: {loss.item()}")  # Should decrease over time
```

### Visualize Embeddings
```python
from sklearn.manifold import TSNE

# Reduce to 2D
embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)

# Plot
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
```

### Monitor Training
```python
# Track metrics
wandb.log({
    "loss": loss.item(),
    "lr": get_lr(optimizer),
    "epoch": epoch
})
```

---

## 📚 Tài Liệu Tham Khảo

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [PhoBERT](https://github.com/VinAIResearch/PhoBERT)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Contrastive Learning](https://arxiv.org/abs/2002.05709)
