import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from itertools import cycle

# Завантаження даних
X = np.loadtxt('data_clustering.txt', delimiter=',')

# Оцінка ширини вікна для X
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# Кластеризація даних методом зсуву середнього
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Отримання центрів та кількості кластерів
cluster_centers = meanshift_model.cluster_centers_
print(f"Координати центрів кластерів:\n{cluster_centers}")

labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print(f"Кількість кластерів = {num_clusters}")

# Візуалізація результатів
plt.figure(figsize=(10, 6))
markers = 'o*dvs'
colors = cycle('bgrcmyk')
for i, marker, col in zip(range(num_clusters), markers, colors):
    cluster_data = X[labels == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                marker=marker, color=col, edgecolor='black', s=70, label=f'Кластер {i+1}')

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', 
            color='black', s=200, label='Центри кластерів')

plt.title('Результат кластеризації')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

# Оцінюємо якість кластеризації
print("\nОцінка якості кластеризації:")
print("Silhouette Score:", round(silhouette_score(X, labels), 3))

