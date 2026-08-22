"""
Amaç:
    - UCI Heart Disease veri setini kullanarak logistic regression modeli ile ikili sınıflandırma problemi çözme
    - Model, bir bireyin kalp hastalığına sahio olup olmadığını tahmin etmeyi amaçlar ve accuracy metriği ile değerlendirilir

Veri seti:
    - UCI Machine learning repo: https://archive.ics.uci.edu/dataset/45/heart+disease
    - veri seti bireylere ait demografik ve klinik ölçümlerini içeriyor
    - features: yaş, cinsiyet, ağrı tipi, kolestrol, kan basıncı vb.
    - hedef değişken
        - 0: hastalık yok
        - 1: hastalık var

Plan/Program:
    1. Veri seti yükle ve temel analizleri yap
    2. Veri seti içerisinde eksik değer kontrolü yap gerekirse temizle
    3. Öznitelik ve hedef değişkenlerin ayrılması
    4. Eğitim ve test veri setlerinin oluşturulması
    5. Logistic regression modelinin tanımlanması ve eğitilmesi
    6. Modelin test veri seti ile değerlendirilmesi

Kurulumlar
pip install pandas scikit-learn ucimlrepo
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Veri seti yükle ve temel analizleri yap
heart_disease = fetch_ucirepo(id = 45)

df = pd.DataFrame(data = heart_disease.data.features)
df["target"] = heart_disease.data.targets
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

print(df.head())

# 2. Veri seti içerisinde eksik değer kontrolü yap gerekirse temizle
if df.isna().any().any():
    df.dropna(inplace=True)
    print("nan değerleri veri setinden çıkardık")
else:
    print("nan değer bulunmuyor")

# 3. Öznitelik ve hedef değişkenlerin ayrılması
X = df.drop(["target"], axis = 1).values # features
y = df.target.values

# 4. Eğitim ve test veri setlerinin oluşturulması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 5. Logistic regression modelinin tanımlanması ve eğitilmesi
log_reg = LogisticRegression(penalty="l2", C = 1, max_iter = 100)
log_reg.fit(X_train, y_train)

# 6. Modelin test veri seti ile değerlendirilmesi
acc = log_reg.score(X_test, y_test)
print(f"Accuracy: {acc}")