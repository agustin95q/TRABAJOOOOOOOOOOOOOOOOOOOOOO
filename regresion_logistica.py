# ==========================================================
# CLASIFICACIÓN - REGRESIÓN LOGÍSTICA
# Dataset: consumo_agua_comunas_70.csv
# ==========================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Cargar dataset
df = pd.read_csv("consumo_agua_comunas_70.csv")

# 2️⃣ Crear variable binaria "Consumo_alto"
umbral = df["Consumo_m3"].mean()
df["Consumo_alto"] = (df["Consumo_m3"] >= umbral).astype(int)

print(f"Promedio de consumo: {umbral:.2f} m³")
print(df["Consumo_alto"].value_counts(), "\n")

# 3️⃣ Seleccionar variables predictoras
X = df[["Poblacion", "Ingreso_promedio", "Temperatura_promedio", "Precipitacion_mm", "Mes"]]
y = df["Consumo_alto"]

# 4️⃣ Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5️⃣ Crear y entrenar modelo
modelo = LogisticRegression(max_iter=500)
modelo.fit(X_train, y_train)

# 6️⃣ Predicciones
y_pred = modelo.predict(X_test)

# 7️⃣ Evaluación
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"✅ Exactitud del modelo: {acc:.2f}")
print("\nMatriz de confusión:\n", cm)
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

# 8️⃣ Visualizar matriz de confusión
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Regresión Logística")
plt.show()
