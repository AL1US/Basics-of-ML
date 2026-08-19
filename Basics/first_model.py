# first_model.py
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Загружаем данные
data = load_diabetes()
X = data.data
y = data.target

# Делим на обучающую и тестовую
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказываем
y_pred = model.predict(X_test)

# Ошибка
error = mean_absolute_error(y_test, y_pred)
print(f"Ошибка модели: {error:.2f}")

# Смотрим веса
print("\n=== ВЕСА (КОЭФФИЦИЕНТЫ) ===")
feature_names = data.feature_names
for name, w in zip(feature_names, model.coef_):
    print(f"{name}: {w:.4f}")
print(f"Смещение (bias): {model.intercept_:.4f}")