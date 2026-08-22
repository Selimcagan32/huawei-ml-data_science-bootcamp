"""
Amaç:
    - Sentetik vir veri oluşturarak K Means ve Agglomerative Clustering algoritmasını uygula
    - K Means küme merkezlerinin (centroid) nasıl oluştuğunu görselleştir
    - Dendrogram ile birleştirme yapısı incele

Veri seti:
    - 300 saples, 2 boyutlu, 4 tane küme 

Plan/program:
    1. veri seti oluşturma
    2. veri noktalarının görselleştirme
    3. K Means modelinin tanımlanması ve eğitimi
    4. Her veri noktasının ait olduğu kümenin belirlenmesi
    5. Centroid ile birlikte K Means sonuçlarının görselleştirilmesi
    6. Aynı veri ile hiyerarşik kümeleme uygulaması
    7. Hiyerarşik kümeleme sonuçlarının görselleştirilmesi 
    8. Dendrogram çizdirme

Kurulumlar
pip install scikit-learn matplotlib scipy
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage, dendrogram

# 1. veri seti oluşturma
X, _ = make_blobs(n_samples=300, centers=5, cluster_std=2, random_state=42)

# 2. veri noktalarının görselleştirme
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s = 50, alpha=0.7, edgecolors="k")
plt.title("Ham veri")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# 3. K Means modelinin tanımlanması ve eğitimi
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)

# 4. Her veri noktasının ait olduğu kümenin belirlenmesi
labels = kmeans.labels_
print(labels)

# 5. Centroid ile birlikte K Means sonuçlarının görselleştirilmesi
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s = 50, c = labels, cmap = "viridis", alpha=0.7, edgecolors="k")

# küme merkezleri
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s = 150, c = "red", marker = "X", label= "centroid")
plt.title("K-Means Kümeleme")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# 6. Aynı veri ile hiyerarşik kümeleme uygulaması

agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
agg_labels = agg.fit_predict(X)

# 7. Hiyerarşik kümeleme sonuçlarının görselleştirilmesi 

plt.figure()
plt.scatter(X[:, 0], X[:, 1], s = 50, c = agg_labels, cmap = "viridis", alpha=0.7, edgecolors="k")
plt.title("Agglomerative Kümeleme")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# 8. Dendrogram çizdirme
linked = linkage(X, method="ward")
plt.figure()
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Veri noktaları")
plt.ylabel("Uzaklık")
plt.show()