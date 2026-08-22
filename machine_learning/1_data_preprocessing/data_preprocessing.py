"""
Makine öğrenmesi veri ön işleme pratikleri

Amaç:
    1. Eksik veri tespiti, çıkartılması ve uygun değerler ile doldurma
    2. IQR yöntemiyle sayısal sütunlardaki aykırı değerleri tespit etmek
    3. Kategorik verileri label encoding ve one-hot encoding ile dönüştür
    4. Veriyi train, validasyon ve test kümelerine ayır
    5. Sayısal özelliklere standardization ve normalization uygula

Kurulum:
pip install pandas scikit-learn
pip install -r requirements.txt
"""

# 1. gerekkli kütüphanelerin içeriye aktarılması
import pandas as pd

from sklearn.model_selection import train_test_split # eğitim ve test veri seti oluşturur
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler 

# 2. veri setinin yüklenmesi
df = pd.read_csv("musteri_verisi_ml_pratik.csv")

print(df.head())
print(df.info())

# 3. eksik veri analizi
print(df.isnull().sum())

df_dropna = df.dropna() # eksik veri çıkart
print(f"Eksik veriler çıktıktan sonra: \n{df_dropna}")

df_filled = df.copy()

sayisal_sutunlar = ["yas", "maas", "deneyim_yili"]

# sayısal sütunları medyan ile doldur
for sutun in sayisal_sutunlar:
    medyan_degeri = df_filled[sutun].median()
    df_filled[sutun] = df_filled[sutun].fillna(medyan_degeri)

# kategorik sütunları en sık tekrar eden değer ile doldur
df_filled["egitim"] = df_filled["egitim"].fillna(df_filled["egitim"].mode()[0])

print(f"Eksik veriler doldurulduktan sonra: \n{df_filled}")

# 4. IQR yöntemiyle aykırı değerleri tespit etme

aykiri_deger_maskesi = pd.Series(False, index = df_filled.index)

for sutun in sayisal_sutunlar:

    q1 = df_filled[sutun].quantile(0.25)
    q3 = df_filled[sutun].quantile(0.75)

    iqr = q3 - q1

    alt_sinir = q1 - 1.5 * iqr
    ust_sinir = q3 + 1.5 * iqr

    sutun_maskesi = (
        (df_filled[sutun] < alt_sinir) | (df_filled[sutun] > ust_sinir)
    )

    aykiri_deger_maskesi = aykiri_deger_maskesi | sutun_maskesi

    print(f"Aykırı değer sayısı: {sutun_maskesi.sum()}")

    if sutun_maskesi.any():
        print(f"Aykırı değerler: \n{df_filled.loc[sutun_maskesi, sutun]}")

print(f"En az bir aykırı değer içeren satırlar \n{df_filled.loc[aykiri_deger_maskesi]}")

# aykırı değer içeren satırları veri setinden çıkartalım
df_clean = df_filled.loc[~aykiri_deger_maskesi].copy()
df_clean.reset_index(drop=True, inplace=True)

print(f"Aykırı değerler çıktıktan sonra \n{df_clean}")

# 5. label encoding ve one-hot encoding 

label_encoder = LabelEncoder()

# hedef değişkeni sayısal hale getir
y = label_encoder.fit_transform(df_clean["satin_aldi"])

print(f"Hedef değişken sınıfları: \n {label_encoder.classes_}")
print(y)

# hedef sütunu veri setinden çıkart
X = df_clean.drop(columns=["satin_aldi"])

X = pd.get_dummies(X, columns=["egitim"], drop_first=True, dtype=int)

print(f"Kategorik dönüşüm sonrası özellikler:  \n{X}")

# 6. Veriyi train validasyon ve test kümelerine ayır
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # val = %80, test = %20

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.4, random_state=42, stratify=y_train_val)

print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

# 7. sayısal özelliklerde standardization 

standard_scaler = StandardScaler()

X_train_standard = X_train.copy()
X_val_standard = X_val.copy()
X_test_standard = X_test.copy()

# ölçekleyiciyi yalnızca eğitim verisi üzerinde öğretiyoruz
X_train_standard[sayisal_sutunlar] = (
    standard_scaler.fit_transform(
        X_train[sayisal_sutunlar]
    )
)

# validasyon ve test verilerinde yalnızca transform uygula
X_val_standard[sayisal_sutunlar] = (
    standard_scaler.transform(
        X_val[sayisal_sutunlar]
    )
)

X_test_standard[sayisal_sutunlar] = (
    standard_scaler.transform(
        X_test[sayisal_sutunlar]
    )
)

print(f"X_train_standard: \n{X_train_standard}")

# 8. normalizasyon

minmax_scaler = MinMaxScaler()

X_train_normalized = X_train.copy()
X_val_normalized = X_val.copy()
X_test_normalized = X_test.copy()

# ölçekleyiciyi yalnızca eğitim verisi üzerinde öğretiyoruz
X_train_normalized[sayisal_sutunlar] = (
    minmax_scaler.fit_transform(
        X_train[sayisal_sutunlar]
    )
)

# validasyon ve test verilerinde yalnızca transform uygula
X_val_normalized[sayisal_sutunlar] = (
    minmax_scaler.transform(
        X_val[sayisal_sutunlar]
    )
)

X_test_normalized[sayisal_sutunlar] = (
    minmax_scaler.transform(
        X_test[sayisal_sutunlar]
    )
)

print(f"X_train_normalized: \n{X_train_normalized}")