import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup and Load Data
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# 2. Data Balance (Check samples per category)
df["category"].value_counts().plot(kind="bar", color="steelblue", title="Data Balance")
plt.show()

# 3. Explore Single Columns (Step-by-step like the video)
set_df = df[df["set"] == 1]

# Vẽ trục Y của Set 1 để xem tín hiệu thô
plt.plot(set_df["acc_y"].reset_index(drop=True))
plt.title("Raw Signal - Acc Y (Set 1)")
plt.show()

# 4. Plot All Activities (Looping to see differences)
labels = df["category"].unique()
for label in labels:
    subset = df[df["category"] == label].head(100)
    fig, ax = plt.subplots()
    ax.plot(subset["acc_y"].reset_index(drop=True), label=label)
    ax.set_title(f"Activity Pattern: {label.upper()}")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Acc Y")
    ax.legend()
    plt.show()

# 5. Compare Participants (User Differences)
activity = "wlk"
participant_df = df.query(f"category == '{activity}'").sort_values("user_id")

fig, ax = plt.subplots()
participant_df.groupby("user_id")["acc_y"].head(100).plot(ax=ax)
plt.title(f"Comparison of all Users - Activity: {activity}")
plt.xlabel("Samples")
plt.ylabel("Acc Y")
plt.legend(title="User ID", bbox_to_anchor=(1.0, 1.0))
plt.show()

# 6. Multiple Axes Plot (x, y, z correlation)
label = "jog"
user = 1
multi_axis_df = df.query(f"category == '{label}' and user_id == {user}").head(100).reset_index(drop=True)

fig, ax = plt.subplots()
multi_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
plt.title(f"3-Axis Acceleration - {label.upper()} (User {user})")
plt.show()

# 7. Distribution and Correlation (Statistics)
corr_df = df[["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]].corr()
sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Sensor Correlation Heatmap")
plt.show()

sns.boxplot(x="category", y="acc_y", data=df)
plt.title("Acceleration Intensity per Category")
plt.show()

# 8. Mass Export Figures (Combining Acc and Gyro)
os.makedirs("../../reports/figures/", exist_ok=True)
test_users = df["user_id"].unique()[:3] # Test trước với 3 người dùng đầu tiên

for label in labels:
    for user in test_users:
        subset = df.query(f"category == '{label}' and user_id == {user}").reset_index(drop=True)
        if subset.empty: continue

        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
        
        subset[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
        ax[0].set_title(f"{label.upper()} - User {user}")
        ax[0].set_ylabel("Acc")

        subset[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])
        ax[1].set_ylabel("Gyro")
        ax[1].set_xlabel("Samples")

        plt.savefig(f"../../reports/figures/{label}_{user}.png", dpi=150)
        plt.close(fig)