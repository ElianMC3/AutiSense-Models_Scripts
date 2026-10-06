import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report
)
from imblearn.over_sampling import SMOTE 
import joblib

# =====================================================================
# 1. DEFINICIÓN DINÁMICA DE RUTAS ABSOLUTAS
# =====================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

x_path = os.path.join(script_dir, "X_gaze.npy")
y_path = os.path.join(script_dir, "y_gaze.npy")

if not os.path.exists(x_path) or not os.path.exists(y_path):
    raise FileNotFoundError("Error: Asegúrate de correr primero 'preprocess_gaze.py'.")

print("\n" + "="*60)
print("     === SYSTEM CONTROL: PIPELINE DE ATENCIÓN VISUAL ===")
print("="*60)

# Cargar las matrices de mirada
X = np.load(x_path)
y = np.load(y_path)

print(f" -> [INFO] Registros totales cargados (X): {X.shape}")
print(f" -> [INFO] Distribución real cruda: {np.bincount(y)[0]} Típicos (0) vs {np.bincount(y)[1]} Riesgo Visual (1)\n")

# =====================================================================
# 2. DIVISIÓN DE DATOS ESTRATIFICADA (80% Train, 20% Val)
# =====================================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================================
# 3. APLICACIÓN DE SMOTE PARA EL BALANCEO DE DATOS
# =====================================================================
print(" [!] Iniciando Balanceo de Muestras mediante Algoritmo SMOTE...")
print(f"     • Distribución Original Train -> Clase 0: {np.bincount(y_train)[0]} | Clase 1: {np.bincount(y_train)[1]}")

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"     -> Nueva Distribución Balanceada -> Clase 0: {np.bincount(y_train_res)[0]} | Clase 1: {np.bincount(y_train_res)[1]}")
print("-"*60 + "\n")

# =====================================================================
# 4. INSTANCIAR Y ENTRENAR EL RANDOM FOREST
# =====================================================================
print(" [!] Entrenando Random Forest Classifier (n_estimators=100)...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_res, y_train_res)

# =====================================================================
# 5. EVALUACIÓN Y MÉTRICAS CLINICAS
# =====================================================================
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

print("\n" + "="*22 + " MATRIZ DE CONFUSIÓN " + "="*21)
matriz = confusion_matrix(y_val, y_pred)
print(f"   [ {matriz[0][0]:<4} {matriz[0][1]:<4} ]  -> [ Verdaderos Típicos  |  Falsos Riesgos ]")
print(f"   [ {matriz[1][0]:<4} {matriz[1][1]:<4} ]  -> [ Falsos Típicos      |  Verdaderos Riesgos ]")

print("\n" + "="*18 + " REPORTE DE MÉTRICAS CLÍNICAS " + "="*18)
print(classification_report(y_val, y_pred, target_names=["Típico (0)", "Riesgo Visual (1)"]))

acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_proba)

print("="*62)
print(f" Exactitud Global (Accuracy):   {acc:.4f}  ({acc*100:.1f}%)")
print(f" Precisión Clínicas (Precision): {prec:.4f}  ({prec*100:.1f}%)")
print(f" Sensibilidad Real (Recall):     {rec:.4f}  ({rec*100:.1f}%)")
print(f" F1-Score (Balance Clínico):     {f1:.4f}  ({f1*100:.1f}%)")
print(f" Área bajo la curva (AUC-ROC):   {auc:.4f}")
print("="*62 + "\n")

# =====================================================================
# 6. ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS
# =====================================================================
nombres_features = [
    "Mean_X", "Var_X", "Std_X", "Q1_X", "Q3_X",
    "Mean_Y", "Var_Y", "Std_Y", "Q1_Y", "Q3_Y",
    "Mean_Z", "Var_Z", "Std_Z", "Q1_Z", "Q3_Z"
]
importancias = model.feature_importances_
indices = np.argsort(importancias)[::-1]

print("====== TOP 5 CARACTERÍSTICAS DE LA MIRADA MÁS PREDICTIVAS ======")
for i in range(5):
    print(f" {i+1}. {nombres_features[indices[i]]:<10}: {importancias[indices[i]]:.4f}")
print("="*62 + "\n")

# =====================================================================
# 7. GUARDAR EL MODELO EXPORTADO
# =====================================================================
modelo_out_path = os.path.join(script_dir, "mejor_modelo_mirada.pkl")
joblib.dump(model, modelo_out_path)

print(f" [ÉXITO] Modelo de Producción exportado exitosamente.")
print(f" Ruta del archivo binario: {modelo_out_path}\n")
