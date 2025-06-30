import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Baca data
train = pd.read_csv("train.csv")
train["date"] = pd.to_datetime(train["date"])

# 2. Buat pivot table: tanggal sebagai index, cluster sebagai kolom
pivot_df = train.groupby(["date", "cluster_id"])["electricity_consumption"].mean().reset_index()
pivot_df = pivot_df.pivot(index="date", columns="cluster_id", values="electricity_consumption")

# 3. Hitung korelasi antar cluster berdasarkan pola waktu
correlation = pivot_df.corr()

# 4. Tampilkan matriks korelasi
print("Matriks Korelasi antar Cluster:")
print(correlation)

# 5. Rasio antar korelasi
cluster_mean = train.groupby("cluster_id")["electricity_consumption"].mean()
print(cluster_mean)

max_mean = cluster_mean.max()
min_mean = cluster_mean.min()
ratio = max_mean / min_mean
print(f"Rasio tertinggi terhadap terendah: {ratio:.2f}")

# 6. Visualisasikan sebagai heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0)
plt.title("Korelasi Konsumsi Listrik antar Cluster")
plt.tight_layout()
plt.show()
