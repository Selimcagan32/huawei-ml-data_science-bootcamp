"""
Amaç:
    1. KNN, karar ağacı ve logistic regression için grid search ve random search
    2. Farklı arama yöntemleri ile sonuçları karşılaştır

Adımlar:
    1. Gerekli kütüphaneleri içeriye aktar
    2. Örnek veri setini yükle ve eğitim/test split yap
    3. Her model için hiperparametre arama uzayını tanımla
    4. grid search ile en iyi hiperparametreleri bul
    5. random search ile en iyi hiperparametreleri bul
    6. En iyi modeli test verisi üzerinden değerlendir
    7. Sonuçları özet olarak yazdır

Kurulumlar:
    pip install pandas scikit-learn

"""

# 1. Gerekli kütüphaneleri içeriye aktar
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 2. Örnek veri setini yükle ve eğitim/test split yap
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Her model için hiperparametre arama uzayını tanımla
models_and_params = {
    "KNN":{
        "pipeline":Pipeline([
            ("model", KNeighborsClassifier())
        ]),
        "grid_params": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__metric": ["euclidean", "manhattan"]
        },
        "random_params": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__metric": ["euclidean", "manhattan"]
        }
    },
    "Decision Tree":{
        "pipeline": Pipeline([
            ("model", DecisionTreeClassifier(random_state=42))
        ]),
        "grid_params": {
            "model__max_depth": [2,3,4,5,None],
            "model__min_samples_split": [2,4,6],
            "model__criterion": ["gini", "entropy"]
        },
        "random_params": {
            "model__max_depth": [2,3,4,5,None],
            "model__min_samples_split": [2,4,6],
            "model__criterion": ["gini", "entropy"]
        }
    },
    "Logistic Regression": {
        "pipeline": Pipeline([
            ("model", LogisticRegression(max_iter=200, random_state=42))
        ]),
        "grid_params":{
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l1", "l2"]
        },
        "random_params":{
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l1", "l2"]
        }
    }
}

results = []

# 4. grid search ile en iyi hiperparametreleri bul
# 5. random search ile en iyi hiperparametreleri bul
# 6. En iyi modeli test verisi üzerinden değerlendir

for model_name, item in models_and_params.items():
    grid_search = GridSearchCV(
        estimator=item["pipeline"],
        param_grid = item["grid_params"],
        cv = 3,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    grid_test_score = accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))

    random_search = RandomizedSearchCV(
        estimator=item["pipeline"],
        param_distributions=item["random_params"],
        n_iter=4,
        cv = 3,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    random_test_score = accuracy_score(y_test, random_search.best_estimator_.predict(X_test))

    results.append({
        "model":model_name,
        "yontem": "grid_search",
        "cv en iyi score": round(grid_search.best_score_,2),
        "test score": round(grid_test_score, 2),
        "en iyi parametre seti": str(grid_search.best_params_) 
    })

    results.append({
        "model":model_name,
        "yontem": "random_search",
        "cv en iyi score": round(random_search.best_score_,2),
        "test score": round(random_test_score, 2),
        "en iyi parametre seti": str(random_search.best_params_) 
    })

# 7. Sonuçları özet olarak yazdır
result_df = pd.DataFrame(results)
print(result_df)


"""
                 model         yontem  cv en iyi score  test score                              en iyi parametre seti
0                  KNN    grid_search             0.95        0.94  {'model__metric': 'manhattan', 'model__n_neigh...
1                  KNN  random_search             0.95        0.93  {'model__n_neighbors': 5, 'model__metric': 'ma...
2        Decision Tree    grid_search             0.94        0.94  {'model__criterion': 'gini', 'model__max_depth...
3        Decision Tree  random_search             0.93        0.91  {'model__min_samples_split': 2, 'model__max_de...
4  Logistic Regression    grid_search             0.95        0.96           {'model__C': 10, 'model__penalty': 'l2'}
5  Logistic Regression  random_search             0.95        0.96           {'model__penalty': 'l2', 'model__C': 10}

"""