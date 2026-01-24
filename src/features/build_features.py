import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import pickle

#1.READ DATA.
data = pd.read_pickle("../../data/processed/data_processed.pkl")
train_df = pd.read_pickle("../../data/processed/train.pkl")
val_df = pd.read_pickle("../../data/processed/val.pkl")
test_df = pd.read_pickle("../../data/processed/test.pkl")

#2.FUNCTION.

def safe_corr(a, b, eps=1e-8):
    #eps ~ 0, tránh sài số 0.
    #Tính tương quan giữa các trục acc x-y-z và gyr x-y-z
    #Nếu kết quả ra NaN thì quy về 0 vì nó không có ý nghĩa khi so với trục khác.
    if np.std(a) < eps or np.std(b) < eps:
        return 0.0 #float
    return np.corrcoef(a, b)[0, 1] # return corr(a,b)


def extract_window_features(df, window_size=10):
    #mỗi window là 10 (Mốc là 2s để đánh giá.).
    # Chỉ giữ những feature giúp phân biệt activity tốt nhất
    # -> tránh overfitting

    #Ds cột cần xử lý.
    sensor_cols = [
        'acc_x', 'acc_y', 'acc_z',
        'gyr_x', 'gyr_y', 'gyr_z'
    ]

    features_list = []

    #Duyệt theo từng người, set.
    for (user_id, set_id), group in df.groupby(['user_id', 'set']):
        n_windows = len(group) // window_size 
        #Duyệt qua từng window trong 1 set.
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            window = group.iloc[start:end]

            #Ko đủ mẫu thì bỏ qua -> tránh lỗi tính toán.
            if len(window) < window_size:
                continue
            
            #Dict.
            window_features = {
                'user_id': user_id,
                'set': set_id,
                'label': window['label'].iloc[0],
                'weight': window['weight'].iloc[0],
                'height': window['height'].iloc[0],
                'age': window['age'].iloc[0],
                'gender': window['gender'].iloc[0]
            }

            #Duyệt qua từng cột -> trích suất đặc trưng (4 cột đại diện) -> tránh overfitting.
            #Đánh giá riêng lẻ trên x,y,z -> Chuyển động theo hướng nào.
            for col in sensor_cols:
                values = window[col].values

                window_features[f'{col}_mean'] = np.mean(values) # vị trí trung tâm. -> giúp phân biệt hướng.
                window_features[f'{col}_std'] = np.std(values) # độ biến động  -> giúp phân biệt tĩnh/ động
                window_features[f'{col}_rms'] = np.sqrt(np.mean(values ** 2)) #cường độ thật -> đo cường độ.
                window_features[f'{col}_energy'] = np.mean(values ** 2) #công suất trong 2s.

            # Gộp 3 trục -> giúp ko phụ thuộc vào hướng đặt điện thoại.
            acc_mag = np.sqrt(
                window['acc_x']**2 +
                window['acc_y']**2 +
                window['acc_z']**2
            )

            gyr_mag = np.sqrt(
                window['gyr_x']**2 +
                window['gyr_y']**2 +
                window['gyr_z']**2
            )

            # Đánh giá magnitude -> Chuyển động mạnh như nào? (Tránh TH xoay điện thoại).
            window_features['acc_mag_mean'] = np.mean(acc_mag)
            window_features['acc_mag_std'] = np.std(acc_mag)
            window_features['acc_mag_rms'] = np.sqrt(np.mean(acc_mag ** 2))

            window_features['gyr_mag_mean'] = np.mean(gyr_mag)
            window_features['gyr_mag_std'] = np.std(gyr_mag)
            window_features['gyr_mag_rms'] = np.sqrt(np.mean(gyr_mag ** 2))

            features_list.append(window_features) 
    #Những feature trên giúp dự đoán xem trong 2s chuyển động đó là gì (chạy,đi bộ,...)
    return pd.DataFrame(features_list)
#OUTCOME: Hàm safe_corr() là để tính tương quan giữa các trục x-y-z của acc và gyr(chỉ 2 đơn vị này).
#         Hàm extract_window_features, lấy 10 mẫu cho mỗi lần -> giảm tgian tính toán feature -> sài std, mean, RSM, energy:
#         Tính 2 lần: 1 lần là cho từng trục riêng lẻ, 1 lần là tổng quát cả 3 trục (ko phụ thuộc vào hướng như riêng lẻ) giúp phân biệt rõ hơn. 


#EXTRACT FEATURE (DATA TRANSFORM).
print("\nExtracting features from windows...")
WINDOW_SIZE = 10  # 10 samples × 200ms = 2 seconds window

features_data = extract_window_features(data, window_size=WINDOW_SIZE)
features_train = extract_window_features(train_df, window_size=WINDOW_SIZE)
features_val = extract_window_features(val_df, window_size=WINDOW_SIZE)
features_test = extract_window_features(test_df, window_size=WINDOW_SIZE)

#Biến đổi data với hàm ở trên (data transform).


#10 dòng dữ liệu -> 1 dòng đặc trưng.
# print(f"\nFeatures extracted:")
# print(f"All data: {features_data.shape}")
# print(f"Train: {features_train.shape}")
# print(f"Val: {features_val.shape}")
# print(f"Test: {features_test.shape}")

#HANDLE MISSING VALUE.(DATA CLEANING.)

# Check for NaN and infinity
print(f"NaN values in train: {features_train.isna().sum().sum()}")
print(f"Inf values in train: {np.isinf(features_train.select_dtypes(include=[np.number])).sum().sum()}")

# Replace infinity with NaN, then fill with median
features_data = features_data.replace([np.inf, -np.inf], np.nan)
features_train = features_train.replace([np.inf, -np.inf], np.nan)
features_val = features_val.replace([np.inf, -np.inf], np.nan)
features_test = features_test.replace([np.inf, -np.inf], np.nan)

#Không lấy những cột này.
metadata_cols = ['user_id', 'set', 'label', 'weight', 'height', 'age', 'gender']

#danh sách feature.
feature_cols = [col for col in features_train.columns if col not in metadata_cols]

# Impute using training-set medians to avoid leakage/shift
imputer = SimpleImputer(strategy='median') #Median ít ảnh hưởng bởi outlier.
imputer.fit(features_train[feature_cols])

#transform tất cả về cùng 1 chuẩn.
features_data[feature_cols] = imputer.transform(features_data[feature_cols])
features_train[feature_cols] = imputer.transform(features_train[feature_cols])
features_val[feature_cols] = imputer.transform(features_val[feature_cols])
features_test[feature_cols] = imputer.transform(features_test[feature_cols])
#OUTCOME: Check các giá trị NaN và Inf -> thay thế sử dụng median (trung vị) -> fit data với tập train
# -> sau đó transform với các tất cả các tập (data, train ,val, test).


#FEATURE SCALING.
print(f"Number of features to scale: {len(feature_cols)}")

# use scaler (đỡ nhạy với outlier hơn.)
scaler = StandardScaler()
scaler.fit(features_train[feature_cols])

# Transform all datasets
features_data[feature_cols] = scaler.transform(features_data[feature_cols])
features_train[feature_cols] = scaler.transform(features_train[feature_cols])
features_val[feature_cols] = scaler.transform(features_val[feature_cols])
features_test[feature_cols] = scaler.transform(features_test[feature_cols])
#OUTCOME: Đưa hết tất cả về 1 thang đo -> tránh khi training model bị sai lệch.


#DATA VALIDATION.(CHECK LẦN CUỐI TRƯỚC KHI EXPORT).
# Check for remaining NaN or infinity
assert features_train.isna().sum().sum() == 0, "Training data contains NaN!"
assert features_val.isna().sum().sum() == 0, "Validation data contains NaN!"
assert features_test.isna().sum().sum() == 0, "Test data contains NaN!"

assert np.isinf(features_train.select_dtypes(include=[np.number])).sum().sum() == 0, "Training data contains infinity!"
assert np.isinf(features_val.select_dtypes(include=[np.number])).sum().sum() == 0, "Validation data contains infinity!"
assert np.isinf(features_test.select_dtypes(include=[np.number])).sum().sum() == 0, "Test data contains infinity!"
#OUTCOME: sài Assert để check nếu có lỗi là sẽ dừng ngay.



# Check label distribution
print("\nLabel distribution:")
print(f"Train: {features_train['label'].value_counts().sort_index()}")
print(f"Val: {features_val['label'].value_counts().sort_index()}")
print(f"Test: {features_test['label'].value_counts().sort_index()}")
#OUTCOME: Nhìn ra feature nào chiếm nhiều, ít.., 


#Export feature.
print("\nExporting features...")

output_dir = "../../data/processed/"
os.makedirs(output_dir, exist_ok=True)

# Save feature datasets
features_data.to_pickle(os.path.join(output_dir, "features_all.pkl"))
features_train.to_pickle(os.path.join(output_dir, "features_train.pkl"))
features_val.to_pickle(os.path.join(output_dir, "features_val.pkl"))
features_test.to_pickle(os.path.join(output_dir, "features_test.pkl"))

# Save scaler
with open(os.path.join(output_dir, "scaler.pkl"), 'wb') as f:
    pickle.dump(scaler, f)

# Save imputer
with open(os.path.join(output_dir, "imputer.pkl"), 'wb') as f:
    pickle.dump(imputer, f)

# Save feature names for reference
with open(os.path.join(output_dir, "feature_names.txt"), 'w') as f:
    f.write("Metadata columns:\n")
    for col in metadata_cols:
        f.write(f"  {col}\n")
    f.write(f"\nFeature columns ({len(feature_cols)}):\n")
    for col in feature_cols:
        f.write(f"  {col}\n")
#Summary
print(f"Total features created: {len(feature_cols)}")
print(f"Window size: {WINDOW_SIZE} samples (2 seconds)")
print(f"\nDataset sizes:")
print(f"  Train: {features_train.shape[0]} samples")
print(f"  Val:   {features_val.shape[0]} samples")
print(f"  Test:  {features_test.shape[0]} samples")
print(f"  Total: {features_data.shape[0]} samples")
#OUTCOME: Trích suất file cho phần training model.

import matplotlib.pyplot as plt

label_counts = features_train['label'].value_counts().sort_index()

plt.figure()
plt.bar(label_counts.index, label_counts.values)
plt.xlabel("Activity label")
plt.ylabel("Number of windows")
plt.title("Label Distribution after Windowing (Train set)")
plt.show()


plt.figure()

for label in features_train['label'].unique():
    subset = features_train[features_train['label'] == label]
    plt.scatter(
        subset['acc_mag_rms'],
        subset['gyr_mag_rms'],
        label=label,
        alpha=0.5
    )

plt.xlabel("Acceleration RMS")
plt.ylabel("Gyroscope RMS")
plt.title("Movement Intensity by Activity")
plt.legend()
plt.show()



import numpy as np

selected_features = [
    'acc_x_mean', 'acc_y_mean', 'acc_z_mean',
    'acc_mag_mean', 'acc_mag_std',
    'gyr_mag_mean', 'gyr_mag_std'
]

corr = features_train[selected_features].corr()

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(selected_features)), selected_features, rotation=45)
plt.yticks(range(len(selected_features)), selected_features)
plt.title("Feature Correlation Matrix")
plt.show()