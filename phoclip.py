# Cấu hình mô hình và dữ liệu
import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoTokenizer, AutoModel
import py_vncorenlp
import albumentations as A
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Tải VnCoreNLP và cấu hình phân đoạn từ
py_vncorenlp.download_model()
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"])

# Cấu hình tham số
class CFG:
    debug = False
    image_path = "/home/nhannv/Hello/AI_Ngoc_Dung/TrinhHao/images"
    batch_size = 64
    num_workers = 8
    epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    max_length = 70
    temperature = 1.0
    size = 224
    num_projection_layers = 1
    projection_dim = 512
    dropout = 0.1

# Định nghĩa mô hình và tokenizer
__text_models__ = {
    "PhoBERT-large": "vinai/phobert-large"
}
__image_models__ = {
    "ResNet50": "resnet50"
}
CFG.text_encoder_model = __text_models__["PhoBERT-large"]
CFG.text_tokenizer = CFG.text_encoder_model
CFG.text_embedding = 1024
CFG.segmenter = lambda sentence: ' '.join(rdrsegmenter.word_segment(sentence))
CFG.image_encoder_model = __image_models__["ResNet50"]
CFG.image_embedding = 2048
CFG.image_encoder_trainable = False
CFG.text_encoder_trainable = True
CFG.image_encoder_pretrained = True
CFG.text_encoder_pretrained = True

# Xử lý dữ liệu ảnh
def move_images(source, destination, prefix=""):
    os.makedirs(destination, exist_ok=True)
    files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
    copied_count = 0
    skipped_count = 0
    with ThreadPoolExecutor() as executor:
        def copy_one(file):
            nonlocal copied_count, skipped_count
            src = os.path.join(source, file)
            dst = os.path.join(destination, prefix + file)
            if os.path.exists(dst):
                skipped_count += 1
                return
            shutil.copy2(src, dst)
            copied_count += 1
        list(tqdm(executor.map(copy_one, files), total=len(files), desc="Copying"))
    print(f"Copied: {copied_count}, Skipped: {skipped_count}")

# Tạo thư mục và tải dữ liệu
os.makedirs("tmp/images", exist_ok=True)
base_dir = '/home/nhannv/Hello/AI_Ngoc_Dung/TrinhHao'

# Xử lý dữ liệu COCO
COCO_PREFIX_TRAIN = "COCO_train2014_"
COCO_PREFIX_VAL = "COCO_val2014_"
SRC_TRAIN = f"{base_dir}/train2014"
SRC_VAL = f"{base_dir}/val2014"
DEST_DIR = "images"
move_images(SRC_TRAIN, DEST_DIR, COCO_PREFIX_TRAIN)
move_images(SRC_VAL, DEST_DIR, COCO_PREFIX_VAL)

# Tạo file JSONL từ COCO
def create_jsonl(input_path, output_path, prefix):
    with open(input_path, 'r') as f:
        data = json.load(f)
    image_dict = {img['id']: img['file_name'] for img in data['images']}
    with jsonlines.open(output_path, 'w') as writer:
        for ann in data['annotations']:
            file_name = image_dict.get(ann['image_id'])
            if file_name:
                writer.write({'image': f"{prefix}{str(ann['image_id']).zfill(12)}.jpg", 'caption': ann['caption']})

create_jsonl(f"{base_dir}/captions_train2014.json", f"{base_dir}/cocopath_train.jsonl", COCO_PREFIX_TRAIN)
create_jsonl(f"{base_dir}/captions_val2014.json", f"{base_dir}/cocopath_val.jsonl", COCO_PREFIX_VAL)

# Tải dữ liệu Flickr, KTVIC, OpenViIC
def load_jsonl(path):
    imgs, caps = [], []
    with jsonlines.open(path) as reader:
        for line in reader:
            imgs.append(line['image'])
            caps.append(line['caption'])
    return pd.DataFrame({'image': imgs, 'caption': caps})

df_coco_train = load_jsonl(f"{base_dir}/cocopath_train.jsonl")
df_coco_val = load_jsonl(f"{base_dir}/cocopath_val.jsonl")
df_coco = pd.concat([df_coco_train, df_coco_val], ignore_index=True)

# Flickr
os.makedirs("flickr", exist_ok=True)
FLICKR_PREFIX = "flickr-"
with open(f"{base_dir}/flickr/captions_vi.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
image_list = [FLICKR_PREFIX + line.strip().split('\t')[0] for line in lines]
caption_list = [line.strip().split('\t')[1] for line in lines]
df_flickr = pd.DataFrame({'image': image_list, 'caption': caption_list})
move_images(f"{base_dir}/flickr/Images", f"{base_dir}/images", FLICKR_PREFIX)

# KTVIC
os.makedirs("ktvic", exist_ok=True)
KTVIC_PREFIX = "ktvic-"
for file in ['train_data.json', 'test_data.json']:
    with open(f"{base_dir}/ktvic/{file}", 'r', encoding='utf-8') as f:
        data = json.load(f)
    images = [{'id': item['id'], 'image': KTVIC_PREFIX + item['filename']} for item in data['images']]
    captions = [{'id': item['image_id'], 'caption': item['caption']} for item in data['annotations']]
    images_df = pd.DataFrame(images)
    captions_df = pd.DataFrame(captions)
    df_temp = pd.merge(images_df, captions_df, on='id').drop(['id'], axis=1)
    if file == 'train_data.json':
        df_ktvic = df_temp
    else:
        df_ktvic_val = df_temp
move_images(f"{base_dir}/ktvic/train-images", f"{base_dir}/images", KTVIC_PREFIX)
move_images(f"{base_dir}/ktvic/public-test-images", f"{base_dir}/images", KTVIC_PREFIX)

# OpenViIC
os.makedirs("open-ViIC", exist_ok=True)
OPENVIIC_PREFIX = "openviic-"
for file in ['uit-openviic-annotation-train.json', 'uit-openviic-annotation-dev.json', 'uit-openviic-annotation-test.json']:
    with open(f"{base_dir}/open-ViIC/{file}", 'r') as f:
        json_data = json.load(f)
    data = [{"image": OPENVIIC_PREFIX + image_path, "caption": caption} for image_path, image_data in json_data.items() for caption in image_data["captions"]]
    df_temp = pd.DataFrame(data)
    if file == 'uit-openviic-annotation-train.json':
        df_openviic = df_temp
    elif file == 'uit-openviic-annotation-dev.json':
        df_openviic_val = df_temp
    else:
        df_openviic_test = df_temp
move_images(f"{base_dir}/open-ViIC/images", f"{base_dir}/images", OPENVIIC_PREFIX)

# Gộp dữ liệu
data_df = pd.concat([df_flickr, df_ktvic, df_openviic, df_coco], ignore_index=True)
valid_df = pd.concat([df_ktvic_val, df_openviic_val, df_coco_val], ignore_index=True)

# Kiểm tra ảnh thiếu
missing = [img for img in data_df['image'] if not os.path.exists(os.path.join('images', img))]
print(f"Total images: {len(data_df)}, Missing: {len(missing)}")

# Định nghĩa dataset và dataloader
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        self.image_filenames = image_filenames
        self.captions = [CFG.segmenter(caption) for caption in captions]
        self.encoded_captions = tokenizer(self.captions, padding=True, truncation=True, max_length=CFG.max_length)
        self.transforms = transforms
    def __len__(self):
        return len(self.captions)
    def __getitem__(self, idx):
        try:
            image_path = os.path.join(CFG.image_path, self.image_filenames[idx])
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transforms(image=image)['image']
            item = {key: torch.tensor(values[idx]) for key, values in self.encoded_captions.items()}
            item['image'] = torch.tensor(image).permute(2, 0, 1).float()
            item['caption'] = self.captions[idx]
            return item
        except:
            new_idx = (idx + 1) % len(self.image_filenames)
            return self.__getitem__(new_idx)

def get_transforms(mode="train"):
    return A.Compose([
        A.Resize(CFG.size, CFG.size, always_apply=True),
        A.Normalize(max_pixel_value=255.0, always_apply=True),
    ])

# Định nghĩa mô hình
class ImageEncoder(nn.Module):
    def __init__(self, model_name=CFG.image_encoder_model, pretrained=CFG.image_encoder_pretrained, trainable=CFG.image_encoder_trainable):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        for p in self.model.parameters():
            p.requires_grad = trainable
    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.text_encoder_pretrained, trainable=CFG.text_encoder_trainable):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name) if pretrained else AutoModel(config=AutoConfig.from_pretrained(model_name))
        for p in self.model.parameters():
            p.requires_grad = trainable
        self.target_token_idx = 0
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=CFG.projection_dim, dropout=CFG.dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CLIPModel(nn.Module):
    def __init__(self, temperature=CFG.temperature, image_embedding=CFG.image_embedding, text_embedding=CFG.text_embedding):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature
    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax((images_similarity + texts_similarity) / 2 * self.temperature, dim=-1)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        return (images_loss + texts_loss).mean() / 2.0

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    return loss if reduction == "none" else loss.mean()

# Hàm huấn luyện và đánh giá
def make_train_valid_dfs():
    return data_df, valid_df

def build_loaders(dataframe, tokenizer, mode):
    dataset = CLIPDataset(dataframe["image"].values, dataframe["caption"].values, tokenizer, get_transforms(mode))
    return torch.utils.data.DataLoader(dataset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=(mode == "train"))

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    for batch in tqdm(train_loader, total=len(train_loader)):
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()
        loss_meter.update(loss.item(), batch["image"].size(0))
    return loss_meter

def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    for batch in tqdm(valid_loader, total=len(valid_loader)):
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        loss_meter.update(loss.item(), batch["image"].size(0))
    return loss_meter

def main():
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)
    train_df, valid_df = make_train_valid_dfs()
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(model.image_projection.parameters(), model.text_projection.parameters()), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=CFG.patience, factor=CFG.factor)
    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, "epoch")
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")
        lr_scheduler.step(valid_loss.avg)

# Hàm sinh embedding
def get_image_embeddings(valid_df, model_path):
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Extracting image embeddings"):
            imgs = batch["image"].to(CFG.device, non_blocking=True)
            feats = model.image_encoder(imgs)
            emb = model.image_projection(feats)
            all_embeds.append(emb.cpu())
    torch.cuda.empty_cache()
    return model, torch.cat(all_embeds, dim=0)

def get_text_embeddings(valid_df, model_path):
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Extracting text embeddings"):
            ids = batch["input_ids"].to(CFG.device, non_blocking=True)
            mask = batch["attention_mask"].to(CFG.device, non_blocking=True)
            feats = model.text_encoder(input_ids=ids, attention_mask=mask)
            emb = model.text_projection(feats)
            all_embeds.append(emb.cpu())
    torch.cuda.empty_cache()
    return model, torch.cat(all_embeds, dim=0)

# Hàm tìm kiếm và đánh giá
def find_matches(model, database_embeddings, image_filenames, text=None, image_path=None, n=25):
    model.eval()
    query_embeddings_n = None
    if text:
        text = CFG.segmenter(text)
        tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)
        encoded_text = tokenizer([text], padding=True, truncation=True, max_length=CFG.max_length)
        batch = {k: torch.tensor(v).to(CFG.device) for k, v in encoded_text.items() if k != "token_type_ids"}
        with torch.no_grad():
            text_features = model.text_encoder(**batch)
            query_embeddings_n = F.normalize(model.text_projection(text_features), p=2, dim=-1)
    if image_path:
        image_tensor = get_tensor_from_path(image_path)
        with torch.no_grad():
            image_features = model.image_encoder(image_tensor.unsqueeze(0).to(CFG.device))
            query_embeddings_n = F.normalize(model.image_projection(image_features), p=2, dim=-1)
    database_embeddings_n = F.normalize(database_embeddings, p=2, dim=-1)
    dot_similarity = torch.matmul(query_embeddings_n, database_embeddings_n.T)
    values, indices = torch.topk(dot_similarity.squeeze(0), n)
    matches = [image_filenames[idx] for idx in indices.tolist()]
    rows = cols = int(n**0.5)
    _, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        path = os.path.join(CFG.image_path, match)
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        ax.axis("off")
    plt.suptitle(f"Top-{n} matches for: {text or os.path.basename(image_path)}", fontsize=12)
    plt.tight_layout()
    plt.show()

def eval_accuracy(text_embeddings, image_embeddings, df, k=10, batch_size=512):
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    correct_predictions = 0
    for start_idx in range(0, len(text_embeddings), batch_size):
        end_idx = min(start_idx + batch_size, len(text_embeddings))
        sim = torch.matmul(text_embeddings[start_idx:end_idx], image_embeddings.T)
        _, topk_indices = sim.topk(k, dim=1)
        for i, indices in enumerate(topk_indices):
            pred_image = df.iloc[start_idx + i]['image']
            top_truths = df.iloc[indices.cpu()]['image'].tolist()
            if pred_image in top_truths:
                correct_predictions += 1
    return correct_predictions / len(text_embeddings)

# Chạy huấn luyện và đánh giá
if __name__ == "__main__":
    main()
    model, image_embeddings = get_image_embeddings(valid_df, "best.pt")
    _, text_embeddings = get_text_embeddings(valid_df, "best.pt")
    top5_accuracy = eval_accuracy(text_embeddings, image_embeddings, valid_df, k=5)
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    find_matches(model, image_embeddings, valid_df['image'].values, text="xe hơi đậu trước ngôi nhà", n=25)