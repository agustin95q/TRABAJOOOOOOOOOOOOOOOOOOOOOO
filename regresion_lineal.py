# ==========================================================
# PREDICCIÓN DE CONSUMO DE AGUA POR COMUNA (REGRESIÓN LINEAL)
# Entrenamiento: años 2022–2023
# Prueba: año 2024
# ==========================================================

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1️⃣ Cargar dataset extendido
df = pd.read_csv("consumo_agua_comunas_70.csv")

print("Vista general del dataset:\n", df.head(), "\n")
print("Años disponibles:", df["Año"].unique(), "\n")

# 2️⃣ Separar en entrenamiento (2022–2023) y prueba (2024)
train = df[df["Año"] < 2024]
test = df[df["Año"] == 2024]

X_train = train[["Poblacion", "Ingreso_promedio", "Temperatura_promedio", "Precipitacion_mm"]]
y_train = train["Consumo_m3"]

X_test = test[["Poblacion", "Ingreso_promedio", "Temperatura_promedio", "Precipitacion_mm"]]
y_test = test["Consumo_m3"]

# 3️⃣ Crear y entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 4️⃣ Predecir sobre datos de prueba (año 2024)
y_pred = modelo.predict(X_test)

# 5️⃣ Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Resultados del Modelo (Predicción 2024):")
print(f"Error Medio Absoluto (MAE): {mae:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.3f}\n")

# 6️⃣ Comparar valores reales vs predichos (muestra)
comparacion = pd.DataFrame({
    "Comuna": test["Comuna"].values,
    "Año": test["Año"].values,
    "Mes": test["Mes"].values,
    "Real": y_test.values,
    "Predicho": y_pred.round(2)
})

print("Ejemplo de comparación real vs predicho (año 2024):\n")
print(comparacion.head(10))



# 8️⃣ Mostrar coeficientes del modelo
coeficientes = pd.DataFrame({
    "Variable": X_train.columns,
    "Coeficiente": modelo.coef_.round(4)
})
print("\nCoeficientes del modelo:\n", coeficientes)
