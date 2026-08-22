"""
Amaç:
    1. Göğüs kanseri veri seti üzerinde bir sınıflandırma modeli eğitmek ve model tahminlerini LIME ve SHAP yöntemleri ile açıklamak

Plan/program:
    1. Gerekli kütüphanelerin yüklenmesi
    2. veri setinin yüklenmesi
    3. verinin eğitim ve test olarak ayrılması
    4. verinin ölçeklenmesi
    5. Random forest modelinin eğitilmesi
    6. Model performansının değerlendirilmesi
    7. LIME ile tek bir tahmin açıklama
    8. SHAP ile özellik katkılarının incelenmesi

Kurulumlar:
pip install pandas numpy matplotlib scikit-learn lime shap
"""
# 1. Gerekli kütüphanelerin yüklenmesi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from lime.lime_tabular import LimeTabularExplainer
import shap

# 2. veri setinin yüklenmesi
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name = "target")

class_names = data.target_names

print(f"veri boyutu: {X.shape}")
print(f"Sınıfılar: {class_names}")

# 3. verinin eğitim ve test olarak ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"eğitim veri boyutu: {X_train.shape}")
print(f"test veri boyutu: {X_test.shape}")

# 4. verinin ölçeklenmesi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index = X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index = X_test.index)

# 5. Random forest modelinin eğitilmesi
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Model performansının değerlendirilmesi
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"accuracy: {accuracy}")

# 7. LIME ile tek bir tahmin açıklama
sample_index = 0

sample = X_test_scaled[sample_index]
sample_original = X_test.iloc[sample_index]

prediction = model.predict([sample])[0]
prediction_proba = model.predict_proba([sample])[0]

print(f"Açıklanacak örnek: \n{sample_original}")

print(f"model tahmini: {class_names[prediction]}")
print(f"Tahmin olasılıkları: {prediction_proba}")

lime_explainer = LimeTabularExplainer(training_data=X_train_scaled, feature_names=X.columns.tolist(), class_names=class_names.tolist(), mode="classification")

lime_exp = lime_explainer.explain_instance(data_row=sample, predict_fn=model.predict_proba, num_features=10)

print("LIME Açıklaması:")

for feature, contribution in lime_exp.as_list():
    print(f"{feature}: {contribution}")

"""
LIME Açıklaması:
worst concave points > 0.73: -0.14387362050514776
worst area > 0.30: -0.13808374239635665
worst radius > 0.50: -0.10441437376892679
worst perimeter > 0.55: -0.08389082553243657
area error > 0.13: -0.055585815638970276
worst texture > 0.60: -0.05241760823379397
mean perimeter > 0.50: -0.050485083314
"""

# 8. SHAP ile özellik katkılarının incelenmesi
shap_explainer = shap.TreeExplainer(model)
shap_values = shap_explainer.shap_values(X_test_scaled)

if isinstance(shap_values, list):
    shap_values_class_1 = shap_values[1]
    expected_value_class_1 = shap_explainer.expected_value[1]
else:
    shap_values_class_1 = shap_values[:, :, 1]
    expected_value_class_1 = shap_explainer.expected_value[1]    

shap.summary_plot(shap_values_class_1, X_test_scaled_df)
plt.show()