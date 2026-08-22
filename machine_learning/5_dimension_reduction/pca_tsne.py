"""
Amaç:
    - iris veri seti üzerinde pca ive t-sne boyut indirgeme yöntemlerini uygulamak
    - 4 boyuttan 2 boyuta indirgeme yapalım

Plan/program:
    1. iris veri setinin yüklenmesi
    2. özellik ve hedef değişkenlerin ayrılması
    3. verilerin standartlaştırılması
    4. PCA modelinin 2 bileşen ile tanımlanması
    5. PCA dönüşümü uygula
    6. PCA sonucunun 2 boyutlu görselleştirilmesi
    7. t-SNE modelinin 2 bileşen ile tanımlanması
    8. t-SNE dönüşümü uygula
    9. t-SNE sonucunun 2 boyutlu görselleştirilmesi

Kurulumlar:
pip install scikit-learn matplotlib
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. iris veri setinin yüklenmesi
iris = load_iris()

# 2. özellik ve hedef değişkenlerin ayrılması
X = iris.data
y = iris.target

# 3. verilerin standartlaştırılması
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. PCA modelinin 2 bileşen ile tanımlanması
pca = PCA(n_components=2)

# 5. PCA dönüşümü uygula
X_pca = pca.fit_transform(X_scaled)
print(X_scaled)
print()
print(X_pca)

# 6. PCA sonucunun 2 boyutlu görselleştirilmesi
plt.figure()

for i in range(len(iris.target_names)):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label = iris.target_names[i])

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA ile iris veri setinin görselleştirilmesi")
plt.legend()
plt.show()

# 7. t-SNE modelinin 2 bileşen ile tanımlanması
tsne = TSNE(n_components=2)

# 8. t-SNE dönüşümü uygula
X_tsne = tsne.fit_transform(X_scaled)

# 9. t-SNE sonucunun 2 boyutlu görselleştirilmesi
plt.figure()

for i in range(len(iris.target_names)):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label = iris.target_names[i])

plt.xlabel("X_tsne 1")
plt.ylabel("X_tsne 2")
plt.title("X_tsne ile iris veri setinin görselleştirilmesi")
plt.legend()
plt.show()