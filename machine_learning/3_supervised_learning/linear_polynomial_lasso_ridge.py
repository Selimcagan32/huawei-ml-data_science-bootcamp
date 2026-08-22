"""
Regresyon modelleri pratiği

Amaç:
    1. Sentetik veri seti oluşturma
    2. Doğrusal, polinomal, Lasso ve Ridge regresyon modellerini uygulama
    3. Lasso feature selection yapalım

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Sentetik veri seti oluştur
    3. Oluşturulan sentetik veriyi görselleştir
    4. Veriyi eğitim ve test olarak ayır
    5. Doğrusal, polinomal, Lasso ve Ridge regresyon modellerini oluştur
    6. Modelleri eğit (training) ve tahmin (prediction) yap
    7. Modellerin performansını karşılaştır
    8. Lasso ile feature selection

Kurulumlar
pip install numpy matplotlib scikit-learn
"""

# 1. Gerekli kütüphanelerin içeriye aktarılması
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 2. Sentetik veri seti oluştur
np.random.seed(42)

# 4 features
x1 = np.random.uniform(0, 10, 200)
x2 = np.random.uniform(0, 10, 200)
x3 = np.random.uniform(0, 10, 200)
x4 = np.random.uniform(0, 10, 200)

X = np.column_stack([x1, x2, x3, x4]) # bağımsız değişken
print(X)

y = ( # x1 ve x2 önemli features, x3 ve x4 önemsiz features
    4
    + 2.5 * x1
    + 1.8 * x2
    + 0.15 * (x1 ** 2)
    - 0.1 * (x2 ** 2)
    + 0.2 * x1 * x2
    + np.random.normal(0, 2, 200)
)

# 3. Oluşturulan sentetik veriyi görselleştir
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x1,x2,y)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
plt.title("Sentetik Veri")
plt.show()

# 4. Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)

# 5. Doğrusal, polinomal, Lasso ve Ridge regresyon modellerini oluştur
linear_model = LinearRegression()

polynomial_model = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("linear", LinearRegression())
    ]
)

lasso_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=0.1))
    ]
)

ridge_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("lasso", Ridge(alpha=0.1))
    ]
)

# 6. Modelleri eğit (training) ve tahmin (prediction) yap
linear_model.fit(X_train, y_train)
polynomial_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)
y_pred_polynomial= polynomial_model.predict(X_test)
y_pred_lasso = lasso_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)

# 7. Modellerin performansını karşılaştır
print(f"Doğrusal: mse: {mean_squared_error(y_test, y_pred_linear)}  --- R2: {r2_score(y_test, y_pred_linear)}")
print(f"Polynomial: mse: {mean_squared_error(y_test, y_pred_polynomial)}  --- R2: {r2_score(y_test, y_pred_polynomial)}")
print(f"Lasso: mse: {mean_squared_error(y_test, y_pred_lasso)}  --- R2: {r2_score(y_test, y_pred_lasso)}")
print(f"Ridge: mse: {mean_squared_error(y_test, y_pred_ridge)}  --- R2: {r2_score(y_test, y_pred_ridge)}")

# 8. Lasso ile feature selection
ozellik_isimleri = np.array(["x1", "x2", "x3", "x4"])
lasso_katsayilari = lasso_model.named_steps["lasso"].coef_

for isim, katsayi in zip(ozellik_isimleri, lasso_katsayilari):
    print(f"{isim}: {katsayi}")

"""
Doğrusal: mse: 6.823888208928172  --- R2: 0.9738263277798985
Polynomial: mse: 3.449543221020774  --- R2: 0.9867689489024832
Lasso: mse: 6.964916341571237  --- R2: 0.9732854009644822
Ridge: mse: 6.836142572651741  --- R2: 0.9737793250023696
x1: 14.64489829381634
x2: 4.6709928066883455
x3: 0.0
x4: -0.0

y = a0 + 14x1 + 4.6x2 + 0x3 + 0x4
"""