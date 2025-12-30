# 12-20-2025
# dataloader
# 1. 현재 설정 (Current Config)
#    - Batch Size: 2 (GPU 메모리 안전을 위해 최소화)
#    - Image Shape: (64, 64, 64) -> 전처리 완료된 상태
#    - Split Ratio: 8:1:1 (Patient-wise Group Split)
#    - Normalization: Z-score (Mean 0, Std 1)

# 2. 다음 실험 시 변경 고려 사항 (Next To-Do)
#    - [ ] Batch Size: 4 또는 8로 상향 (학습 안정성 향상)
#    - [ ] Data Augmentation: 3D Rotation이나 Flip 추가 (과적합 방지)
#    - [ ] Tabular Preprocessing: 현재는 단순 Mapping, 향후 정규화(Min-Max) 고려
#    - [ ] Balancing: AD/MCI/CN 클래스 불균형이 심할 경우 WeightedSampler 도입

# 3. 주의사항
#    - Image Data ID가 없는 행은 Cleaning 과정에서 자동 삭제됨.
#    - -1, -4 등 결측치 코드가 포함된 행도 삭제 처리됨.

import os
import glob
import pandas as pd
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

class ADNI_Multimodal_Dataset(Dataset):
    def __init__(self, dataframe, img_dir):
        self.df = dataframe
        self.img_dir = img_dir
        # 사용할 표 데이터 변수 10개
        self.tab_cols = [
            "Sex", "Age", "PTEDUCAT", "PTMARRY", "APOE4", 
            "VSWEIGHT", "VSBPSYS", "VSBPDIA", "VSPULSE", "VSRESP"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Image Loading (파일명 끝의 I{ID} 매칭)
        img_id = str(int(row["Image Data ID"]))
        search_pattern = os.path.join(self.img_dir, f"*I{img_id}.nii.gz")
        found_files = glob.glob(search_pattern)

        if not found_files:
            raise FileNotFoundError(f"Image ID {img_id} not found in {self.img_dir}")
        
        img_data = nib.load(found_files[0]).get_fdata()
        img = torch.from_numpy(img_data).float().unsqueeze(0) # [1, 64, 64, 64]

        # Intensity Normalization (Z-score)
        img = (img - img.mean()) / (img.std() + 1e-8)

        # 2. Tabular Data Preprocessing
        temp_tab = row[self.tab_cols].copy()
        temp_tab['Sex'] = 1.0 if temp_tab['Sex'] == 'F' else 0.0
        
        marry_map = {'Married': 0, 'Divorced': 1, 'Widowed': 2, 'Never married': 3, 'Unknown': 4}
        if isinstance(temp_tab['PTMARRY'], str):
            temp_tab['PTMARRY'] = marry_map.get(temp_tab['PTMARRY'], 4)

        tab_data = torch.tensor(temp_tab.values.astype(np.float32))

        # 3. Label (0: CN, 1: MCI, 2: AD)
        label = torch.tensor(int(row["DX2"]), dtype=torch.long)

        return {"image": img, "tabular": tab_data, "label": label}

def prepare_adni_loaders(csv_path, img_dir, batch_size=2):
    df = pd.read_csv(csv_path)
    
    # 1. 필수 컬럼 및 Subject 컬럼 결측치 제거
    cols_to_check = ["DX2", "Sex", "Age", "Image Data ID", "Subject"]
    df = df.dropna(subset=cols_to_check)
    
    # 2. Patient-wise Split (Subject 기준 8:1:1)
    # 같은 Subject(환자)는 항상 같은 세트(Train/Val/Test)에 묶임
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, temp_idx = next(splitter.split(df, groups=df['Subject']))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    
    val_test_splitter = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(val_test_splitter.split(temp_df, groups=temp_df['Subject']))
    
    # 3. 데이터로더 생성
    train_ds = ADNI_Multimodal_Dataset(train_df, img_dir)
    val_ds = ADNI_Multimodal_Dataset(temp_df.iloc[val_idx], img_dir)
    test_ds = ADNI_Multimodal_Dataset(temp_df.iloc[test_idx], img_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    CSV = "/home/ads_sry/royseo/projects/adni-mamba/0_data/ADNI-smoke-test-list-wise-deletion-rs_copy.csv"
    IMG = "/home/ads_sry/royseo/data"
    
    t_loader, v_loader, s_loader = prepare_adni_loaders(CSV, IMG, batch_size=2)
    for b in t_loader:
        print(f"✅ 배치 로드 성공! 이미지 모양: {b['image'].shape}")
        print(f"✅ 라벨 데이터: {b['label']}")
        print(f"✅ 표 데이터 모양: {b['tabular'].shape}")
        break