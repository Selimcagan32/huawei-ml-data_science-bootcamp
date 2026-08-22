"""
Musteri Ayrilma (Churn) Tahmini - Ara Odev

Amac:
    - Musteri verileri uzerinden musterinin ayrilip ayrilmayacagini (churn)
      tahmin eden bir siniflandirma modeli gelistirmek
    - Veri okuma, on isleme, oznitelik uretme, train/val/test bolme,
      model egitimi ve degerlendirme adimlarini tek bir dosyada uygulamak

Kullanilan kutuphaneler:
    - pandas: veri okuma ve inceleme
    - scikit-learn: on isleme (encoding, scaling), model egitimi ve metrikler

Calistirma adimlari:
    1. musteri_churn_verisi.csv dosyasi bu dosya ile ayni klasorde olmali
    2. pip install -r requirements.txt
    3. python churn_tahmini.py
"""

# 1. gerekli kutuphanelerin iceriye aktarilmasi
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# 2. veri setinin yuklenmesi
df = pd.read_csv("musteri_churn_verisi.csv")

# 3. veri setinin ilk incelemesi
print(df.head())
print(f"Satir sayisi: {df.shape[0]}, Sutun sayisi: {df.shape[1]}")
print(f"Hedef degisken dagilimi: \n{df['churn'].value_counts()}")

# 4. eksik veri kontrolu
print(f"Eksik deger sayilari: \n{df.isnull().sum()}")

sayisal_sutunlar = ["yas", "gelir", "abonelik_suresi", "destek_talebi_sayisi"]

# sayisal sutunlari medyan ile doldur
for sutun in sayisal_sutunlar:
    medyan_degeri = df[sutun].median()
    df[sutun] = df[sutun].fillna(medyan_degeri)

# kategorik sutunu en sik tekrar eden deger ile doldur
df["sehir"] = df["sehir"].fillna(df["sehir"].mode()[0])

print(f"Doldurma sonrasi eksik deger sayilari: \n{df.isnull().sum()}")

# 5. basit bir oznitelik uretme (feature engineering)
# destek talebinde bulunmus mu bulunmamis mi bilgisini ikili bir degiskene cevirelim
df["destek_talebi_var_mi"] = (df["destek_talebi_sayisi"] > 0).astype(int)

print(df[["destek_talebi_sayisi", "destek_talebi_var_mi"]].head())

# 6. kategorik degiskenleri one-hot encoding ile sayisal forma cevirme
y = df["churn"]
X = df.drop(columns=["churn"])

X = pd.get_dummies(X, columns=["sehir", "uyelik_tipi"], drop_first=True, dtype=int)

print(f"One-hot encoding sonrasi ozellikler: \n{X.head()}")

# 7. veriyi train, validation ve test kumelerine ayirma (stratify ile)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)  # train_val = %80, test = %20

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)  # train = %60, val = %20, test = %20

print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

# 8. sayisal ozelliklerde olcekleme (standardization)
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

# olcekleyiciyi yalnizca egitim verisi uzerinde ogretiyoruz
X_train_scaled[sayisal_sutunlar] = scaler.fit_transform(X_train[sayisal_sutunlar])

# validasyon ve test verilerinde yalnizca transform uyguluyoruz
X_val_scaled[sayisal_sutunlar] = scaler.transform(X_val[sayisal_sutunlar])
X_test_scaled[sayisal_sutunlar] = scaler.transform(X_test[sayisal_sutunlar])

# 9. modellerin tanimlanmasi ve egitilmesi
log_reg = LogisticRegression(max_iter=1000, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)  # bonus model

log_reg.fit(X_train_scaled, y_train)
knn.fit(X_train_scaled, y_train)
tree_clf.fit(X_train_scaled, y_train)

# 10. validasyon verisi uzerinde model karsilastirmasi
modeller = {
    "Logistic Regression": log_reg,
    "KNN": knn,
    "Decision Tree": tree_clf,
}

print("Validasyon sonuclari:")
for isim, model in modeller.items():
    val_pred = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"{isim} - validation accuracy: {val_acc:.3f}")

# en yuksek validasyon accuracy'sine sahip modeli sec
en_iyi_isim = max(
    modeller, key=lambda isim: accuracy_score(y_val, modeller[isim].predict(X_val_scaled))
)
en_iyi_model = modeller[en_iyi_isim]

print(f"\nSecilen model: {en_iyi_isim}")

# 11. secilen modelin test verisi ile degerlendirilmesi
y_test_pred = en_iyi_model.predict(X_test_scaled)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

print(f"\nTest sonuclari ({en_iyi_isim}):")
print(f"Confusion matrix: \n{test_conf_matrix}")
print(f"Accuracy: {test_accuracy:.3f}")
print(f"Precision: {test_precision:.3f}")
print(f"Recall: {test_recall:.3f}")
print(f"F1-score: {test_f1:.3f}")

# 12. kisa yorum
print(f"""
Yorum:
{en_iyi_isim} modeli, validasyon verisi uzerinde en yuksek accuracy'yi verdigi
icin secildi ve test verisi uzerinde {test_accuracy:.3f} accuracy elde etti.
Veri setindeki churn=1 sinifinin churn=0 sinifina gore daha az sayida olmasi
(dengesiz siniflar) nedeniyle precision ve recall degerleri accuracy'den
daha dusuk cikabiliyor; bu yuzden sadece accuracy'ye degil precision, recall
ve f1-score degerlerine birlikte bakmak gerekiyor.
""")
