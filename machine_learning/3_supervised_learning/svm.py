"""
Amaç:
    - Digits veri seti kullanarak SVM ile birlikte çok sınıflı bir sınıflandırma problemi çözelim

Veri seti:
    - digits veri seti 0-9 arasında ki rakamları temsil eden 8x8 boyutunda gri sevide görütülerden oluşur
    - 1797 adet sample var
    - 8x8 pikselden her bir örnekte 64 features var

Plan/program:
    1. Veri setinin yüklenmesi ve temel bilgilerin incelenmesi
    2. örnek görüntülerin görselleştirilmesi
    3. özellik ve hedef değişkenlerin ayrılması
    4. eğitim ve test veri setlerinin oluşturulması
    5. svm modeli oluşturma
    6. modelin eğitilmesi
    7. test verisi üzerinde tahmin yapılması
    8. model performansının sınıflandırma raporu ile değerlendirilmesi

Kurulumlar:
pip install matplotlib scikit-learn
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 1. Veri setinin yüklenmesi ve temel bilgilerin incelenmesi
digits = load_digits()
print(digits.DESCR)

# 2. örnek görüntülerin görselleştirilmesi
fig, axes = plt.subplots(nrows=2, ncols=5, figsize = (8,5), subplot_kw={"xticks": [], "yticks": []})
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = "binary", interpolation = "nearest")
    ax.set_title(f"Label: {digits.target[i]}")

plt.tight_layout()
plt.show()

# 3. özellik ve hedef değişkenlerin ayrılması
X = digits.data
y = digits.target

# 4. eğitim ve test veri setlerinin oluşturulması
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# 5. svm modeli oluşturma
svm = SVC(kernel="linear", random_state=42)

# 6. modelin eğitilmesi
svm.fit(X_train, y_train)

# 7. test verisi üzerinde tahmin yapılması
y_pred = svm.predict(X_test)

# 8. model performansının sınıflandırma raporu ile değerlendirilmesi
cls_report = classification_report(y_test, y_pred)
print(cls_report)

"""
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        33
           1       0.97      1.00      0.98        28
           2       1.00      1.00      1.00        33
           3       0.97      0.94      0.96        34
           4       0.98      0.98      0.98        46
           5       0.96      1.00      0.98        47
           6       1.00      1.00      1.00        35
           7       0.97      0.97      0.97        34
           8       1.00      0.97      0.98        30
           9       0.95      0.93      0.94        40

    accuracy                           0.98       360
   macro avg       0.98      0.98      0.98       360
weighted avg       0.98      0.98      0.98       360
"""