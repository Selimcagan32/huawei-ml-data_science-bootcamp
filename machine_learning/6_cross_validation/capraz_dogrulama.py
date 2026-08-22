"""
Amaç:
    1. K-fold, stratified K-fold ve leave-one-out yöntemlerinin uygulanması
    2. Bu üç yöntemin model değerlendirme mantığını sade ve karşılaştırmalı olarak göster

Adımlar:
    1. gerekli kütüphanelerin içeriye aktarılması
    2. örnek veri setini yükle
    3. basit bir sınıflandırma modeli tanımla
    4. K-fold ile çapraz doğrulama yapalım
    5. Stratified K-fold ile çapraz doğrulama yapılması
    6. Leave-one-out ile çapraz doğrulama yapılması
    7. Sonuçların birlikte yazdırılması

Kurulumlar:
pip install scikit-learn numpy
"""

# 1. gerekli kütüphanelerin içeriye aktarılması
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_score

# 2. örnek veri setini yükle
X, y = load_iris(return_X_y=True)

# 3. basit bir sınıflandırma modeli tanımla
model = LogisticRegression(max_iter=200)

# 4. K-fold ile çapraz doğrulama yapalım
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_accuracy = cross_val_score(model, X, y, cv = kfold, scoring="accuracy")

# 5. Stratified K-fold ile çapraz doğrulama yapılması
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_kfold_accuracy = cross_val_score(model, X, y, cv = stratified_kfold, scoring="accuracy")

# 6. Leave-one-out ile çapraz doğrulama yapılması
loo = LeaveOneOut()
loo_accuracy = cross_val_score(model, X, y, cv = loo, scoring="accuracy")

# 7. Sonuçların birlikte yazdırılması
print(f"kfold: {kfold_accuracy}")
print(f"stratified_kfold: {stratified_kfold_accuracy}")
print(f"loo_accuracy: {loo_accuracy}")

"""
kfold: 
[1.         1.         0.93333333 0.96666667 0.96666667]
stratified_kfold: 
[1.         0.96666667 0.93333333 1.         0.93333333]
loo_accuracy: 
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1.
 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1.]
"""
print("kfold")
print(np.mean(kfold_accuracy))
print(np.std(kfold_accuracy))