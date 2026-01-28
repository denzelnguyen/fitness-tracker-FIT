import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
train_df = pd.read_pickle("../../data/processed/features_train.pkl")
val_df = pd.read_pickle("../../data/processed/features_val.pkl")
test_df = pd.read_pickle("../../data/processed/features_test.pkl")

# 2. Check number of used features
metadata_cols = ['user_id', 'set', 'label', 'weight', 'height', 'age', 'gender']
feature_cols = [col for col in train_df.columns if col not in metadata_cols]

print(f"Number of features used for training: {len(feature_cols)}")

# 3. Split features and labels
X_train = train_df[feature_cols]
y_train = train_df["label"]

X_val = val_df[feature_cols]
y_val = val_df["label"]

X_test = test_df[feature_cols]
y_test = test_df["label"]

# 4. Define model
model = RandomForestClassifier( 
    n_estimators=300, # Số cây tối đa
    random_state=42, # Bảo đảm tất cả các lần chạy đều ra 1 KQ
    max_features = "sqrt", # Mỗi node thường lấy từng này features đã train
    class_weight="balanced", # Vì sự imbalance của các window ta đã thấy ở FE
)

# 5. Train
model.fit(X_train, y_train)

# 6. Evaluation
y_val_pred = model.predict(X_val)
print("\nValidation accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

y_test_pred = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# Test with a single label
LABEL_NAME = {
    0: "Downstairs",
    1: "Jogging",
    2: "Sitting",
    3: "Standing",
    4: "Upstairs",
    5: "Walking"
}

sample = 10  # chọn 1 sample bất kỳ

x_demo = X_test.iloc[sample:sample+1]
y_true = y_test.iloc[sample]

y_pred = model.predict(x_demo)[0]

print("Try with a random label")
print("Prediction:", LABEL_NAME[y_pred])
print("True label:", LABEL_NAME[y_true])