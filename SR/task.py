import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# 1. Завантажуємо та змінюємо дані
articles = pd.read_csv('articles.csv')
articles = articles.drop_duplicates(subset='title', keep='first')
articles['id'] = range(1, len(articles) + 1)
articles = articles[['id', 'title', 'text']]
print("Кількість записів:", len(articles))
print("Поля:", articles.columns.tolist())
print(articles.head(5))
print("-"*50)

articles['all_text'] = articles['title'].astype(str) + ' ' + articles['text'].astype(str)

# 2. Створюємо TF-IDF вектори
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = vectorizer.fit_transform(articles['title'])
# tfidf_matrix = vectorizer.fit_transform(articles['all_text'])

# 3. Кластеризація TF-IDF векторів
n_clusters = 28
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
articles['cluster'] = kmeans.fit_predict(tfidf_matrix)
print(articles['cluster'].value_counts())
print("-"*50)

# 4. Функція пошуку схожих постів
def search_with_clusters(df, tfidf_matrix, vectorizer, query, top_n=5):
    # визначення вектора TF-IDF для запиту
    query_vector = vectorizer.transform([query])
    
    # визначення найближчого кластера
    cluster_distances = cosine_similarity(query_vector, kmeans.cluster_centers_)
    best_cluster = np.argmax(cluster_distances)
    
    # статті з цього кластера
    cluster_indices = df[df['cluster'] == best_cluster].index
    cluster_pos = df.index.get_indexer(cluster_indices)  # позиції у tfidf_matrix
    cluster_tfidf = tfidf_matrix[cluster_pos, :]
    
    # визначення косинусної схожості
    similarities = cosine_similarity(query_vector, cluster_tfidf).flatten()
    
    # визначення Топ-N позицій у кластері за подібністю
    top_idx = similarities.argsort()[::-1][:top_n]
    
    # отримання правильних індексів постів у DataFrame
    cluster_indices_list = cluster_indices.to_list()
    top_article_indices = [cluster_indices_list[i] for i in top_idx]
    
    # отримання топ-статей
    top_articles = df.loc[top_article_indices].copy()
    top_articles['scores'] = similarities[top_idx]
    return top_articles


# 5. Приклад пошуку
query = "Recurrent Neural Network in TensorFlow"
# query = articles.loc[articles['id'] == 136, 'all_text'].iloc[0]
search_results = search_with_clusters(
    articles, tfidf_matrix, vectorizer, query, top_n=5
)

print("Схожі пости:")
for _, row in search_results.iterrows():
    print(f"ID: {row['id']}")
    print(f"TITLE: {row['title']}")
    print(f"SCORE: {row['scores']:.4f}")
    print(f"CLUSTER: {row['cluster']}")
    print("-"*50)
    print()

# 6. Визначення метрик Silhouette score і Inertia
sil_scores = []
inertia = []
K = range(2, 30)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(tfidf_matrix)
    score = silhouette_score(tfidf_matrix, labels)
    sil_scores.append(score)
    inertia.append(kmeans.inertia_)

plt.plot(K, sil_scores, marker='o')
plt.xlabel("Кількість кластерів")
plt.ylabel("Silhouette score")
plt.title("Silhouette method для KMeans")
plt.show()

best_k_silhouette = K[sil_scores.index(max(sil_scores))]
print(f"Оптимальна кількість кластерів за Silhouette Score: {best_k_silhouette}")

plt.plot(K, inertia, marker='o')
plt.xlabel("Кількість кластерів")
plt.ylabel("Inertia")
plt.title("Elbow method для KMeans")
plt.show()