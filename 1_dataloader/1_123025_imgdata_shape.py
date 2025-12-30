# 12-30-2025
# image data resample 한 후 size & resolution 확인


# 1_12302025_check_multiple_samples.py
import pandas as pd
import os
import glob
import nibabel as nib
import numpy as np

csv_path = "/home/ads_sry/royseo/projects/adni-mamba/0_data/ADNI-smoke-test-list-wise-deletion-rs_copy.csv"
img_dir = "/home/ads_sry/royseo/data"

# 데이터 로드
df = pd.read_csv(csv_path)

# 1. 20개 랜덤 샘플링 (데이터가 20개보다 적으면 전체 선택)
sample_size = min(20, len(df))
df_samples = df.sample(n=sample_size, random_state=42)

print(f"--- 총 {sample_size}개의 랜덤 샘플 검사를 시작합니다 ---")
results = []

for i, (idx, row) in enumerate(df_samples.iterrows()):
    img_id = str(int(row["Image Data ID"]))
    search_pattern = os.path.join(img_dir, f"*I{img_id}.nii.gz")
    found = glob.glob(search_pattern)
    
    if found:
        img_path = found[0]
        nii = nib.load(img_path)
        shape = nii.shape
        # 해상도(Voxel spacing) 가져오기: nii.header.get_zooms()
        # 보통 (x_mm, y_mm, z_mm) 형태입니다.
        zooms = nii.header.get_zooms()
        
        results.append({
            "ID": img_id,
            "Shape": shape,
            "Resolution": tuple(np.round(zooms, 2)),
            "File": os.path.basename(img_path)[:30] + "..." # 이름이 길어서 줄임
        })
        print(f"[{i+1}/{sample_size}] ID {img_id}: OK")
    else:
        print(f"[{i+1}/{sample_size}] ID {img_id}: ❌ 파일을 찾을 수 없음")

# 2. 결과 리포트 (표 형태로 출력)
print("\n" + "="*70)
print(f"{'Image ID':<10} | {'Shape':<15} | {'Voxel Res (mm)':<18} | {'File Name'}")
print("-" * 70)
for res in results:
    print(f"{res['ID']:<10} | {str(res['Shape']):<15} | {str(res['Resolution']):<18} | {res['File']}")
print("="*70)

# 3. 요약 통계
shapes = [r['Shape'] for r in results]
if all(s == (64, 64, 64) for s in shapes):
    print("\n✅ 모든 샘플이 (64, 64, 64)로 일관되게 리샘플링되어 있습니다.")
else:
    print("\n⚠️ 경고: 이미지 크기가 다른 샘플이 발견되었습니다. 전처리를 다시 확인하세요.")