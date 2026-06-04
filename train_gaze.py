import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib  # Para guardar este modelo clásico

# 1. CARGAR LAS MATRICES DE MIRADA
if not os.path.exists("X_gaze.npy") or not os.path.exists("y_gaze.npy"):
    raise FileNotFoundError("Error: Asegúrate de correr primero 'preprocess_gaze.py'.")

X = np.load("X_gaze.npy")
y = np.load("y_gaze.npy")

print("=== BASE DE DATOS DE ATENCIÓN VISUAL CARGADA ===")
print(f" -> Total de registros de sesiones (X): {X.shape}")
print(f" -> Total de etiquetas de comunicación (y): {y.shape}")
print(f" -> Distribución real: {np.bincount(y)[0]} Típicos (0) vs {np.bincount(y)[1]} Riesgo Visual (1)")

# 2. DIVISIÓN DE DATOS ESTRATIFICADA (80% Entrenamiento, 20% Validación)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. INSTANCIAR EL MODELO ENSEMBLE (RANDOM FOREST)
# Activamos 'class_weight=balanced' para obligar al árbol a penalizar fuertemente
# los errores cometidos al intentar identificar a los 284 niños típicos.
model = RandomForestClassifier(
    n_estimators=150,         # 150 árboles de decisión trabajando en paralelo
    max_depth=10,             # Límite de profundidad para evitar sobreajuste
    class_weight="balanced",  # Compensación matemática del desbalanceo
    random_state=42,
    n_jobs=-1                 # Usa todos los núcleos de tu procesador
)

# 4. ENTRENAMIENTO DEL MODELO DE MIRADA
print("\nEntrenando Bosque Aleatorio para análisis de fijación ocular...")
model.fit(X_train, y_train)

# 5. EVALUACIÓN Y MÉTRICAS DE VALIDACIÓN
y_pred = model.predict(X_val)

print("\n================ MATRIZ DE CONFUSIÓN ================")
print(confusion_matrix(y_val, y_pred))

print("\n================ REPORTES DE MÉTRICAS CLINICAS ================")
print(classification_report(y_val, y_pred, target_names=["Típico (0)", "Riesgo Visual (1)"]))

accuracy = accuracy_score(y_val, y_pred)
print(f"Exactitud Global (Accuracy): {accuracy:.4f}")

# 6. ANÁLISIS DE CARACTERÍSTICAS MÁS IMPORTANTES
# Esto te dará argumentos científicos brutales para tus profesores
nombres_features = [
    "Mean_X", "Var_X", "Std_X", "Q1_X", "Q3_X",
    "Mean_Y", "Var_Y", "Std_Y", "Q1_Y", "Q3_Y",
    "Mean_Z", "Var_Z", "Std_Z", "Q1_Z", "Q3_Z"
]
importancias = model.feature_importances_
indices = np.argsort(importancias)[::-1]

print("\n====== TOP 5 CARACTERÍSTICAS DE LA MIRADA MÁS PREDICTIVAS ======")
for i in range(5):
    print(f" {i+1}. {nombres_features[indices[i]]}: {importancias[indices[i]]:.4f}")

# 7. CONGELAR Y GUARDAR EL MODELO
joblib.dump(model, "modelo_fijacion_ocular.pkl")
print("\n¡Éxito! El Algoritmo 2 quedó guardado en la raíz como 'modelo_fijacion_ocular.pkl'")