import os
import json
import torch
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split  # <--- VŨ KHÍ MỚI


def get_data(split_name='train', window_size=100, step_size=50, save_path=None):
    print(f"⏳ Đang tải dataset 'RS2002/WiFall'...")
    hf_dataset = load_dataset("RS2002/WiFall")
    df = hf_dataset[split_name].to_pandas()

    print("⚙️ Đang giải mã chuỗi CSI...")
    df['csi_array'] = df['data'].apply(json.loads)
    raw_csi_matrix = np.stack(df['csi_array'].values)
    raw_labels = df['taget'].values

    le = LabelEncoder()
    encoded_labels = le.fit_transform(raw_labels)

    # 1. TIỀN XỬ LÝ TOÀN BỘ DATA (CHUẨN HÓA + PCA)
    print("🧹 Đang Chuẩn hóa và ép PCA (104 -> 12 kênh)...")
    scaler = StandardScaler()
    scaled_csi = scaler.fit_transform(raw_csi_matrix)

    pca = PCA(n_components=12)
    pca_csi = pca.fit_transform(scaled_csi)

    # 2. CẮT CỬA SỔ TRƯỢT TRÊN TOÀN BỘ TẬP
    print("📏 Đang cắt cửa sổ trượt...")
    X_samples, Y_samples = [], []
    for start in range(0, len(pca_csi) - window_size, step_size):
        end = start + window_size
        window_labels = encoded_labels[start:end]
        if len(set(window_labels)) == 1:
            X_samples.append(pca_csi[start:end, :])
            Y_samples.append(window_labels[0])

    X_all = np.stack(X_samples)
    Y_all = np.array(Y_samples)

    # 3. TRỘN LẪN VÀ CHIA CẮT PHÂN TẦNG (STRATIFIED SPLIT)
    # stratify=Y_all đảm bảo tỷ lệ 50% Fall được giữ nguyên vẹn ở cả 2 tập!
    print("🔀 Đang xáo trộn và chia Train/Test ngẫu nhiên...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_all, Y_all, test_size=0.2, stratify=Y_all, random_state=42
    )

    data_dict = {
        'X_train': torch.tensor(X_train, dtype=torch.float32),
        'Y_train': torch.tensor(Y_train, dtype=torch.long),
        'X_test': torch.tensor(X_test, dtype=torch.float32),
        'Y_test': torch.tensor(Y_test, dtype=torch.long),
        'classes': le.classes_
    }

    print(f"✅ HOÀN TẤT! Train: {data_dict['X_train'].shape}, Test: {data_dict['X_test'].shape}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(data_dict, save_path)

    return data_dict