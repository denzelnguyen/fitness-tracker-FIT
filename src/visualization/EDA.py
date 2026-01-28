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