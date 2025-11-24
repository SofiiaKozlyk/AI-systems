import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Завантажуємо набір даних Iris
iris = load_iris()
X = iris['data']
y = iris['target']

# Ініціалізуємо модель KMeans
# n_clusters=3 — очікуємо 3 класи ірисів
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300,
                tol=0.0001, random_state=0)

# Навчаємо модель на даних
kmeans.fit(X)

# Отримуємо передбачені мітки кластерів
y_kmeans = kmeans.predict(X)

# Отримуємо координати центрів кластерів
centers = kmeans.cluster_centers_

# Візуалізація
plt.figure(figsize=(12, 5))

# Справжні класи
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
plt.title("Справжні класи Iris")
plt.xlabel("Довжина чашолистка")
plt.ylabel("Ширина чашолистка")
for i, name in enumerate(iris['target_names']):
    plt.scatter([], [], color=plt.cm.viridis(i / 2), label=name)
plt.legend()

# Кластери, знайдені методом K-Means
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.6, label='Центри кластерів')
plt.title("Результати кластеризації K-Means")
plt.xlabel("Довжина чашолистка")
plt.ylabel("Ширина чашолистка")
for i, name in enumerate(iris['target_names']):
    plt.scatter([], [], color=plt.cm.viridis(i / 2), label=name)
plt.legend()
plt.tight_layout()
plt.show()

# Порівнюємо кластеризацію з реальними класами
print("Справжні мітки класів (y):")
print(y)
print("Прогнозовані мітки кластерів (y_kmeans):")
print(y_kmeans)

# Оцінюємо якість кластеризації
print("\nОцінка якості кластеризації:")
print("Inertia:", round(kmeans.inertia_, 2))
print("Silhouette Score:", round(silhouette_score(X, y_kmeans), 3))
print("Calinski-Harabasz Index:", round(calinski_harabasz_score(X, y_kmeans), 2))
print("Davies-Bouldin Index:", round(davies_bouldin_score(X, y_kmeans), 2))
