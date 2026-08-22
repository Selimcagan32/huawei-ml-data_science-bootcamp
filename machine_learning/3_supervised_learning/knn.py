"""
Amaç:
    - Göğüs kanseri veri setini kullanarak KNN algoritmasıyla sınıflandırma yapalım
    - Modelin doğruluk oranını hesapla, farklı K değerleri için hiperparametre araması yapalım

Plan/program:
    1. veri setinin yüklenmesi
    2. feature ve hedef değişkenlerin ayrılması
    3. eğitim ve test verilerinin oluşturulması
    4. Özelliklerin ölçeklendirilmesi
    5. KNN eğitimi ve testi
    6. Doğruluk oranı ve confusion matrix
    7. Hiperparametre ayarlaması
    8. sonuçların grafiksel olarak gösterilmesi

Kurulumlar
pip install scikit-learn pandas matplotlib
"""

from sklearn.datasets import load_breast_cancer # dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

import pandas as pd
import matplotlib.pyplot as plt

# 1. veri setinin yüklenmesi
cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target
print(df.head())

# 2. feature ve hedef değişkenlerin ayrılması
X = cancer.data
y = cancer.target 

#  3. eğitim ve test verilerinin oluşturulması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Özelliklerin ölçeklendirilmesi
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. KNN eğitimi ve testi
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)

# 6. Doğruluk oranı ve confusion matrix
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy}")

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"confusion matrix: \n{conf_matrix}")
"""
accuracy: 0.9590643274853801
confusion matrix: 
[[ 59   4]
 [  3 105]]
"""

# 7. Hiperparametre ayarlaması
# 8. sonuçların grafiksel olarak gösterilmesi
k_accuracy = []
k_values = []
for k in range(3, 15):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    k_accuracy.append(accuracy_score(y_pred, y_test))
    k_values.append(k)

plt.plot(k_values, k_accuracy)
plt.show()