import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from jiwer import cer
import matplotlib.pyplot as plt
import seaborn as sns
import math
from data_preprocessing import DataPreprocessor

class CharTokenizer:
    def __init__(self):
        self.specials = ['<pad>', '<sos>', '<eos>']
        self.chars = (self.specials + 
                      list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') +
                      list('àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ') +
                      list('0123456789') +
                      list(' .,?!:;-()[]{}"\'/')) 
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}
        self.pad_id = self.char2idx['<pad>']
        self.sos_id = self.char2idx['<sos>']
        self.eos_id = self.char2idx['<eos>']

    def __call__(self, text):
        return [self.char2idx.get(c, self.char2idx[' ']) for c in text]

    def decode(self, indices):
        chars = []
        for idx in indices:
            if idx == self.eos_id:
                break
            chars.append(self.idx2char.get(idx, ' '))
        return ''.join(chars)

class OCRDataset(Dataset):
    def __init__(self, data_dir, split='train', tokenizer=None, target_size=(85, 515), processed_folder='processed_data_folder'):
        self.split = split
        self.base_dir = os.path.join(data_dir, split)
        self.img_dir = os.path.join(self.base_dir, 'img')
        self.box_dir = os.path.join(self.base_dir, 'box')
        self.tokenizer = tokenizer
        self.target_size = target_size
        self.processed_folder = os.path.join(self.base_dir, processed_folder)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        self.files = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.png'))]
        self.data = []
        self.error_files = []
        
        preprocessor = DataPreprocessor(data_dir, split, tokenizer, target_size, processed_folder=processed_folder)
        preprocessor.validate_csv_files()
        preprocessor.process_data()
        self.error_files.extend(preprocessor.error_files)
        
        print(f"Đang tải dữ liệu đã xử lý từ {self.processed_folder}...")
        for fn in self.files:
            idx = 0
            while True:
                img_processed_path = os.path.join(self.processed_folder, f"{fn.replace('.jpg', f'_box{idx}_processed.png')}")
                text_txt_path = os.path.join(self.processed_folder, f"{fn.replace('.jpg', f'_box{idx}_text.txt')}")
                if not (os.path.exists(img_processed_path) and os.path.exists(text_txt_path)):
                    break
                
                try:
                    img = Image.open(img_processed_path).convert('RGB')
                    if img.size != (515, 85):
                        print(f"Cảnh báo: Hình ảnh {img_processed_path} có kích thước {img.size}, kỳ vọng (515, 85)")
                    img_tensor = self.transform(img)
                    
                    with open(text_txt_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) < 2:
                            print(f"Định dạng file văn bản không hợp lệ: {text_txt_path}")
                            self.error_files.append((text_txt_path, "Định dạng file văn bản không hợp lệ"))
                            idx += 1
                            continue
                        transcript = lines[0].replace('original_text=', '').strip()
                        tokenized_text_str = lines[1].replace('tokenized_text=', '').strip()
                        if not tokenized_text_str:
                            print(f"Văn bản mã hóa rỗng trong {text_txt_path}")
                            self.error_files.append((text_txt_path, "Văn bản mã hóa rỗng"))
                            idx += 1
                            continue
                        tokenized_text = tokenized_text_str.split(',')
                        try:
                            label = torch.tensor([int(x) for x in tokenized_text if x], dtype=torch.long)
                            if len(label) == 0:
                                print(f"Văn bản mã hóa không hợp lệ trong {text_txt_path}: {tokenized_text_str}")
                                self.error_files.append((text_txt_path, f"Văn bản mã hóa không hợp lệ: {tokenized_text_str}"))
                                idx += 1
                                continue
                        except ValueError as e:
                            print(f"Lỗi khi phân tích văn bản mã hóa trong {text_txt_path}: {e}")
                            self.error_files.append((text_txt_path, f"Lỗi khi phân tích văn bản mã hóa: {e}"))
                            idx += 1
                            continue
                    
                    self.data.append((img_tensor, label, transcript, fn, idx))
                except Exception as e:
                    print(f"Lỗi khi tải file {img_processed_path} hoặc {text_txt_path}: {e}")
                    self.error_files.append((img_processed_path, str(e)))
                idx += 1
        
        print(f"Đã tải {len(self.data)} mẫu dữ liệu.")
        if len(self.data) > 0:
            img_tensor, label_tensor, transcript, _, _ = self.data[0]
            print(f"Kích thước tensor hình ảnh mẫu: {img_tensor.shape}")
            print(f"Kích thước tensor nhãn mẫu: {label_tensor.shape}")
            print(f"Văn bản gốc mẫu: {transcript}")
        else:
            print("Cảnh báo: Không tải được dữ liệu. Kiểm tra dữ liệu đầu vào và thư mục đã xử lý.")

        if self.error_files:
            print("\n=== Danh sách các file lỗi ===")
            for file_path, error_msg in self.error_files:
                print(f"File: {file_path}")
                print(f"Lỗi: {error_msg}\n")
        else:
            print("\nKhông có file nào gây lỗi.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    transcripts = [item[2] for item in batch]
    filenames = [item[3] for item in batch]
    box_indices = [item[4] for item in batch]
    return {'images': images, 'labels': labels, 'transcripts': transcripts, 'filenames': filenames, 'box_indices': box_indices}

def preprocess_cropped_image(img, target_size=(85, 515)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size[::-1], interpolation=cv2.INTER_CUBIC)
    img_pil = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img_tensor = transform(img_pil)
    return img_tensor

class CNNBackbone(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.features.children()))
        self.proj = nn.Linear(1280, d_model)
        self.d_model = d_model

    def forward(self, x):
        feat = self.features(x)
        feat = feat.flatten(2).transpose(1, 2)
        feat = self.proj(feat)
        return feat

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, return_attention=False):
        tgt2, self_attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2, cross_attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        if return_attention:
            return tgt, self_attn_weights, cross_attn_weights
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=3, num_heads=4, d_ff=1024, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.transformer_layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model, num_heads, d_ff, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.num_layers = num_layers

    def forward(self, tgt, memory, return_attention=False):
        B, T = tgt.size()
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.pos_encoder[:, :T, :]
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)
        
        output = tgt_emb.transpose(0, 1)
        attn_weights = [] if return_attention else None
        
        for layer in self.transformer_layers:
            if return_attention:
                output, self_attn, cross_attn = layer(output, memory.transpose(0, 1), 
                                                    tgt_mask=tgt_mask, return_attention=True)
                attn_weights.append(cross_attn)
            else:
                output = layer(output, memory.transpose(0, 1), tgt_mask=tgt_mask)
        
        output = output.transpose(0, 1)
        logits = self.fc_out(output)
        
        if return_attention:
            return logits, attn_weights
        return logits

class OCRModel(nn.Module):
    def __init__(self, vocab_size, max_len=100, pad_id=0, sos_id=1, eos_id=2):
        super().__init__()
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.max_len = max_len
        self.encoder = CNNBackbone()
        self.decoder = TransformerDecoder(vocab_size, max_len=max_len)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, images, tgt_seq=None, return_attention=False):
        memory = self.encoder(images)
        
        if tgt_seq is not None:
            if return_attention:
                logits, attn_weights = self.decoder(tgt_seq, memory, return_attention=True)
                loss = self.criterion(logits[:, :-1].reshape(-1, logits.size(-1)), 
                                    tgt_seq[:, 1:].reshape(-1))
                return logits, loss, attn_weights
            else:
                logits = self.decoder(tgt_seq, memory)
                loss = self.criterion(logits[:, :-1].reshape(-1, logits.size(-1)), 
                                    tgt_seq[:, 1:].reshape(-1))
                return logits, loss
        
        B = images.size(0)
        device = images.device
        generated = torch.full((B, 1), self.sos_id, dtype=torch.long, device=device)
        
        for _ in range(self.max_len - 1):
            logits = self.decoder(generated, memory)
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == self.eos_id).all():
                break
        
        if return_attention:
            _, attn_weights = self.decoder(generated, memory, return_attention=True)
            return generated[:, 1:], attn_weights
        return generated[:, 1:]

def visualize_results(image, results, output_path, truth_texts=None, attn_weights=None):
    img = image.copy()
    
    # Tạo figure với 3 subplot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6), dpi=300)  # Tăng kích thước và DPI
    
    # Hình ảnh gốc
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Hình ảnh gốc', fontsize=14)
    ax1.axis('off')
    
    # Vẽ tất cả các hộp giới hạn
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # Màu khác nhau cho các box
    for idx, result in enumerate(results):
        box = result['box']
        text = result['text']
        x_min, y_min, x_max, y_max = box
        
        # Vẽ hộp với màu khác nhau
        color = colors[idx % len(colors)]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 3)  # Độ dày hộp
        
        # Tạo nhãn với văn bản dự đoán và thực tế
        label = f"Dự đoán: {text}"
        if truth_texts and idx < len(truth_texts):
            label += f"\nThực tế: {truth_texts[idx]}"
        
        # Tính toán vị trí nhãn để không chồng lấn
        label_lines = label.split('\n')
        for i, line in enumerate(label_lines):
            cv2.putText(img, line, (x_min, y_min - 30 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)  # Nền đen
            cv2.putText(img, line, (x_min, y_min - 30 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)  # Chữ trắng
    
    # Hình ảnh với các hộp
    ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax2.set_title('Hình ảnh với dự đoán', fontsize=14)
    ax2.axis('off')
    
    # Bản đồ nhiệt attention
    if attn_weights is not None and len(attn_weights) > 0:
        attn = attn_weights[-1].mean(dim=0).cpu().numpy()  # Shape: [tgt_len, src_len]
        h, w = 85, 515
        feat_h, feat_w = int(np.ceil(h / 32)), int(np.ceil(w / 32))  # feat_h = 3, feat_w = 17
        expected_src_len = feat_h * feat_w
        
        if attn.shape[1] == expected_src_len:
            attn = attn.mean(axis=0).reshape(feat_h, feat_w)
            attn_resized = cv2.resize(attn, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            sns.heatmap(attn_resized, cmap='viridis', alpha=0.6, ax=ax3, cbar=True)
            ax3.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), alpha=0.4)
            ax3.set_title('Bản đồ nhiệt Attention', fontsize=14)
            ax3.axis('off')
        else:
            ax3.text(0.5, 0.5, 'Không có bản đồ nhiệt', 
                    horizontalalignment='center', verticalalignment='center', fontsize=12)
            ax3.axis('off')
    else:
        ax3.text(0.5, 0.5, 'Không có trọng số Attention', 
                horizontalalignment='center', verticalalignment='center', fontsize=12)
        ax3.axis('off')
    
    # Lưu hình ảnh với độ phân giải cao
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def infer_image(model, tokenizer, image_path, data_dir, output_path, device='cuda'):
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Đọc và kiểm tra ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể tải hình ảnh tại {image_path}")
    
    # Tìm file box tương ứng
    filename = os.path.basename(image_path)
    box_path = os.path.join(data_dir, 'test', 'box', filename.replace('.jpg', '.csv'))
    if not os.path.exists(box_path):
        raise ValueError(f"Không tìm thấy file box tại {box_path}")
    
    df = pd.read_csv(box_path, quotechar='"', encoding='utf-8')
    if df.empty or not all(col in df for col in ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']):
        raise ValueError(f"Định dạng file box không hợp lệ tại {box_path}")
    
    results = []
    truth_texts = []
    attn_weights_list = []
    
    # Xử lý từng bounding box
    for idx, row in df.iterrows():
        box_coords = [int(row['x1']), int(row['y1']), 
                      int(row['x2']), int(row['y2']),
                      int(row['x3']), int(row['y3']),
                      int(row['x4']), int(row['y4'])]
        x_min, x_max = min(box_coords[::2]), max(box_coords[::2])
        y_min, y_max = min(box_coords[1::2]), max(box_coords[1::2])
        
        # Cắt và tiền xử lý vùng ảnh
        cropped_img = img[y_min:y_max, x_min:x_max]
        if cropped_img.size == 0:
            print(f"Cảnh báo: Hình ảnh cắt rỗng cho hộp {idx} trong {filename}")
            continue
        
        img_tensor = preprocess_cropped_image(cropped_img).unsqueeze(0).to(device)
        
        # Dự đoán
        with torch.no_grad():
            generated, attn_weights = model(img_tensor, return_attention=True)
        
        pred_text = tokenizer.decode(generated[0].cpu().numpy())
        truth_text = row.get('text', '')
        
        results.append({'box': [x_min, y_min, x_max, y_max], 'text': pred_text})
        truth_texts.append(truth_text)
        attn_weights_list.append(attn_weights)
    
    # Vẽ heatmap (sử dụng attention weights của box đầu tiên)
    attn_weights = attn_weights_list[0] if attn_weights_list else None
    visualize_results(img, results, output_path, truth_texts, attn_weights)
    
    return results, truth_texts

def test_ocr(model, tokenizer, data_dir, vis_dir, test_image_paths=None, device='cuda'):
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    os.makedirs(vis_dir, exist_ok=True)
    
    total_cer = 0
    total_samples = 0
    results_summary = []
    
    # Nếu có danh sách ảnh cụ thể
    if test_image_paths:
        image_paths = test_image_paths
    else:
        # Lấy tất cả ảnh trong thư mục test/img
        test_img_dir = os.path.join(data_dir, 'test', 'img')
        image_paths = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) 
                      if f.lower().endswith(('.jpg', '.png'))]
    
    for image_path in tqdm(image_paths, desc="Đang kiểm tra ảnh"):
        try:
            output_path = os.path.join(vis_dir, f"test_{os.path.basename(image_path)}")
            results, truth_texts = infer_image(model, tokenizer, image_path, data_dir, output_path, device)
            
            # Tính CER cho mỗi ảnh
            image_cer = 0
            for res, truth in zip(results, truth_texts):
                pred_text = res['text']
                image_cer += cer(truth, pred_text)
                results_summary.append({
                    'image': os.path.basename(image_path),
                    'box': res['box'],
                    'predicted': pred_text,
                    'truth': truth,
                    'cer': cer(truth, pred_text)
                })
            
            image_cer /= len(results) if results else 1
            total_cer += image_cer
            total_samples += 1
            
            print(f"\nẢnh: {os.path.basename(image_path)}")
            for i, (res, truth) in enumerate(zip(results, truth_texts)):
                print(f"Hộp {i}: Dự đoán = {res['text']}, Thực tế = {truth}, CER = {cer(truth, res['text']):.4f}")
            print(f"CER trung bình của ảnh: {image_cer:.4f}")
            
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {image_path}: {e}")
    
    avg_cer = total_cer / total_samples if total_samples > 0 else float('inf')
    print(f"\nCER trung bình trên toàn bộ tập kiểm tra: {avg_cer:.4f}")
    
    # Lưu tóm tắt kết quả vào CSV
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(os.path.join(vis_dir, 'test_results_summary.csv'), index=False, encoding='utf-8')
        print(f"Tóm tắt kết quả đã được lưu vào {os.path.join(vis_dir, 'test_results_summary.csv')}")
    
    return avg_cer, results_summary

def train_ocr(data_dir, num_epochs=10, batch_size=4, lr=1e-4, checkpoint_dir='checkpoints', vis_dir='visualizations'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    tokenizer = CharTokenizer()
    train_dataset = OCRDataset(data_dir, split='train', tokenizer=tokenizer)
    print(f"Số file huấn luyện: {len(train_dataset.files)}")
    if len(train_dataset) == 0:
        raise ValueError("Tập dữ liệu huấn luyện rỗng. Kiểm tra dữ liệu đầu vào và file CSV.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)

    test_dataset = OCRDataset(data_dir, split='test', tokenizer=tokenizer)
    print(f"Số file kiểm tra: {len(test_dataset.files)}")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)

    model = OCRModel(len(tokenizer.chars)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Huấn luyện"):
            images = batch['images'].to(device)
            labels = batch['labels']
            
            # Padding labels để có độ dài đồng nhất trong batch
            max_len = max(len(l) for l in labels)
            tgt_seq = torch.full((len(labels), max_len), tokenizer.pad_id, dtype=torch.long, device=device)
            for i, l in enumerate(labels):
                tgt_seq[i, :len(l)] = l.to(device)
            
            optimizer.zero_grad()
            _, loss = model(images, tgt_seq)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_cer = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1} Kiểm tra"):
                images = batch['images'].to(device)
                labels = batch['labels']
                transcripts = batch['transcripts']
                
                # Padding labels
                max_len = max(len(l) for l in labels)
                tgt_seq = torch.full((len(labels), max_len), tokenizer.pad_id, dtype=torch.long, device=device)
                for i, l in enumerate(labels):
                    tgt_seq[i, :len(l)] = l.to(device)
                
                _, loss = model(images, tgt_seq)
                val_loss += loss.item()
                
                generated = model(images)
                decoded = [tokenizer.decode(g.cpu().numpy()) for g in generated]
                val_cer += sum(cer(gt, pred) for gt, pred in zip(transcripts, decoded))

        avg_val_loss = val_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
        avg_val_cer = val_cer / len(test_loader) if len(test_loader) > 0 else float('inf')

        print(f"Epoch {epoch+1}/{num_epochs}, Loss huấn luyện: {avg_train_loss:.4f}, "
              f"Loss kiểm tra: {avg_val_loss:.4f}, CER kiểm tra: {avg_val_cer:.4f}")

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))

        scheduler.step(avg_val_loss)

    return tokenizer, model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Huấn luyện và kiểm tra mô hình OCR với trực quan hóa")
    parser.add_argument('--data_dir', default='data1', help='Đường dẫn đến thư mục dữ liệu')
    parser.add_argument('--num_epochs', type=int, default=40, help='Số epoch huấn luyện')
    parser.add_argument('--batch_size', type=int, default=4, help='Kích thước batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Tốc độ học')
    parser.add_argument('--checkpoint_dir', default='checkpoints', help='Thư mục lưu checkpoint')
    parser.add_argument('--vis_dir', default='visualizations', help='Thư mục lưu hình ảnh trực quan')
    parser.add_argument('--infer_image_path', default=None, help='Đường dẫn đến ảnh để suy luận (tùy chọn)')
    parser.add_argument('--test_image_paths', default=None, help='Danh sách đường dẫn ảnh kiểm tra, phân tách bằng dấu phẩy (tùy chọn)')
    args = parser.parse_args()

    print("Bắt đầu huấn luyện mô hình OCR...")
    tokenizer, model = train_ocr(args.data_dir, args.num_epochs, args.batch_size, args.lr, 
                                 args.checkpoint_dir, args.vis_dir)
    
    # Chạy kiểm tra trên tập test hoặc danh sách ảnh cụ thể
    print("\nBắt đầu kiểm tra mô hình...")
    test_image_paths = args.test_image_paths.split(',') if args.test_image_paths else None
    avg_cer, results_summary = test_ocr(model, tokenizer, args.data_dir, args.vis_dir, test_image_paths)
    
    # Nếu có ảnh suy luận riêng
    if args.infer_image_path:
        print(f"\nChạy suy luận trên ảnh: {args.infer_image_path}")
        output_path = os.path.join(args.vis_dir, f"infer_{os.path.basename(args.infer_image_path)}")
        results, truth_texts = infer_image(model, tokenizer, args.infer_image_path, args.data_dir, output_path)
        print("Kết quả suy luận:")
        for i, (res, truth) in enumerate(zip(results, truth_texts)):
            print(f"Hộp {i}: Dự đoán = {res['text']}, Thực tế = {truth}, CER = {cer(truth, res['text']):.4f}")