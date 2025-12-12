import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from torchvision import transforms

class DataPreprocessor:
    def __init__(self, data_dir, split='train', tokenizer=None, target_size=(85, 515), max_length=100, processed_folder='processed_data_folder'):
        self.split = split
        self.base_dir = os.path.join(data_dir, split)
        self.img_dir = os.path.join(self.base_dir, 'img')
        self.box_dir = os.path.join(self.base_dir, 'box')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_size = target_size
        self.processed_folder = os.path.join(self.base_dir, processed_folder)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        os.makedirs(self.processed_folder, exist_ok=True)
        
        self.files = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.png'))]
        self.error_files = []

    def validate_csv_files(self):
        expected_columns = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'text']
        for fn in self.files:
            box_path = os.path.join(self.box_dir, fn.replace('.jpg', '.csv'))
            if not os.path.exists(box_path):
                print(f"CSV file not found: {box_path}")
                self.error_files.append((box_path, "File does not exist"))
                continue
            try:
                temp_df = pd.read_csv(box_path, quotechar='"', encoding='utf-8', nrows=0)
                if list(temp_df.columns) == expected_columns:
                    df = pd.read_csv(box_path, quotechar='"', encoding='utf-8', header=0)
                else:
                    df = pd.read_csv(box_path, quotechar='"', encoding='utf-8',
                                    header=None,
                                    names=expected_columns)
                
                if list(df.columns) != expected_columns:
                    print(f"CSV file {box_path} has incorrect columns: {list(df.columns)}")
                    self.error_files.append((box_path, f"Incorrect columns: {list(df.columns)}"))
                    continue
                
                for _, row in df.iterrows():
                    if pd.isna(row).any() or len(row) != 9:
                        print(f"Invalid data in {box_path}: {row}")
                        self.error_files.append((box_path, f"Invalid data: {row}"))
                        continue
                    try:
                        _ = [int(row['x1']), int(row['y1']), 
                             int(row['x2']), int(row['y2']),
                             int(row['x3']), int(row['y3']),
                             int(row['x4']), int(row['y4'])]
                        _ = str(row['text'])
                    except (ValueError, TypeError) as e:
                        print(f"Error checking row in {box_path}: {row}, Error: {e}")
                        self.error_files.append((box_path, f"Row error: {e}"))
            except Exception as e:
                print(f"Error checking CSV file {box_path}: {e}")
                self.error_files.append((box_path, str(e)))

    def process_data(self):
        processed_images = set(f for f in os.listdir(self.processed_folder) if f.endswith('_processed.png'))
        processed_texts = set(f.replace('_text.txt', '') for f in os.listdir(self.processed_folder) if f.endswith('_text.txt'))
        missing_files = [f for f in self.files if any(f.replace('.jpg', f'_box{i}_processed') not in processed_images or 
                                                     f.replace('.jpg', f'_box{i}_text') not in processed_texts 
                                                     for i in range(100))]
        
        if missing_files:
            print(f"Processing {len(missing_files)} data samples and saving to {self.processed_folder}...")
            for fn in tqdm(missing_files, desc="Processing data"):
                img_path = os.path.join(self.img_dir, fn)
                box_path = os.path.join(self.box_dir, fn.replace('.jpg', '.csv'))
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Cannot read image: {img_path}")
                    self.error_files.append((img_path, "Cannot read image"))
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                boxes = []
                if os.path.exists(box_path):
                    try:
                        temp_df = pd.read_csv(box_path, quotechar='"', encoding='utf-8', nrows=0)
                        if list(temp_df.columns) == ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'text']:
                            df = pd.read_csv(box_path, quotechar='"', encoding='utf-8', header=0)
                        else:
                            df = pd.read_csv(box_path, quotechar='"', encoding='utf-8',
                                            header=None,
                                            names=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'text'])
                        
                        for _, row in df.iterrows():
                            if pd.isna(row).any() or len(row) != 9:
                                print(f"Invalid data in {box_path}: {row}")
                                self.error_files.append((box_path, f"Invalid data: {row}"))
                                continue
                            try:
                                box_coords = [int(row['x1']), int(row['y1']), 
                                             int(row['x2']), int(row['y2']),
                                             int(row['x3']), int(row['y3']),
                                             int(row['x4']), int(row['y4'])]
                                text = str(row['text']) if pd.notna(row['text']) else ''
                                boxes.append((box_coords, text))
                            except (ValueError, TypeError) as e:
                                print(f"Error processing row in {box_path}: {row}, Error: {e}")
                                self.error_files.append((box_path, f"Row error: {row}, {e}"))
                                continue
                    except Exception as e:
                        self.error_files.append((box_path, str(e)))
                        print(f"Error reading CSV file {box_path}: {e}")
                        continue
                
                for idx, (box_coords, transcript) in enumerate(boxes):
                    x_min = min(box_coords[::2])
                    x_max = max(box_coords[::2])
                    y_min = min(box_coords[1::2])
                    y_max = max(box_coords[1::2])
                    
                    cropped_img = img[y_min:y_max, x_min:x_max]
                    if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                        print(f"Invalid crop region for {fn}, box {idx}")
                        continue
                    cropped_img = cv2.resize(cropped_img, self.target_size[::-1], interpolation=cv2.INTER_CUBIC)
                    img_pil = Image.fromarray(cropped_img)
                    img_tensor = self.transform(img_pil)
                    
                    img_array = img_tensor.numpy().transpose(1, 2, 0)
                    img_array = (img_array * 0.5 + 0.5) * 255
                    img_array = img_array.astype(np.uint8)
                    img_processed = Image.fromarray(img_array)
                    img_processed_path = os.path.join(self.processed_folder, f"{fn.replace('.jpg', f'_box{idx}_processed.png')}")
                    img_processed.save(img_processed_path)

                    ids = [self.tokenizer.sos_id] + self.tokenizer(transcript)[:self.max_length-2] + [self.tokenizer.eos_id]
                    if not ids or len(ids) == 2:
                        print(f"Empty or invalid transcript for {fn}, box {idx}: {transcript}")
                        self.error_files.append((img_processed_path, f"Empty or invalid transcript: {transcript}"))
                        continue
                    ids += [self.tokenizer.pad_id] * (self.max_length - len(ids))
                    tokenized_text = ','.join(map(str, ids))

                    text_txt_path = os.path.join(self.processed_folder, f"{fn.replace('.jpg', f'_box{idx}_text.txt')}")
                    with open(text_txt_path, 'w', encoding='utf-8') as f:
                        f.write(f"original_text={transcript}\ntokenized_text={tokenized_text}\nbox_coords={','.join(map(str, box_coords))}")

        if self.error_files:
            print("\n=== List of erroneous files ===")
            for file_path, error_msg in self.error_files:
                print(f"File: {file_path}")
                print(f"Error: {error_msg}\n")
        else:
            print("\nNo files caused errors.")