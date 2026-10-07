# AutiSense - Modelos de Machine Learning y Preprocesamiento

Este repositorio contiene los scripts de procesamiento de datos biométricos y entrenamiento de modelos de Machine Learning del proyecto **AutiSense**, un sistema diseñado para la detección temprana (tamizaje) de señales de riesgo de autismo (TEA) en niños mediante el análisis de la atención visual y movimientos corporales.

---

## Origen de los Datos: Dataset DREAM

El proyecto utiliza el **DREAM Dataset** (*Robot Enhanced Therapy for Children with Autism Spectrum Disorder*), alojado en el **Swedish National Data Service (SND)**[cite: 1, 3].

### Características del Dataset:
* **Dispositivo de Captura:** Cámaras **Kinect (RGBD)** que registran vectores 3D de la mirada (*Gaze*) y posiciones del esqueleto corporal (*Skeleton joints*).
* **Volumen:** Registros extraídos de sesiones clínicas de interacción asistida por tecnología y terapeutas.
* **Muestras Procesadas:** Un total de **3,120 registros de sesiones** con **15 características métricas** extraídas por prueba (promedios, varianzas, desviaciones estándar y cuantiles Q1/Q3 en los ejes X, Y, Z).

---

## Estructura del Repositorio

| Archivo | Descripción |
| :--- | :--- |
| `preprocess_gaze.py` | Extrae y calcula las 15 características estadísticas de fijación ocular a partir de las series de tiempo crudas del dataset DREAM, generando las matrices `X_gaze.npy` y `y_gaze.npy`. |
| `preprocess_all.py` | Pipeline extendido de preprocesamiento que integra tanto las métricas de la mirada (*Gaze*) como las coordenadas del esqueleto (*Moves*). |
| `train_gaze.py` | Script principal de entrenamiento. Utiliza el algoritmo **SMOTE** para corregir el desbalanceo de clases y entrena un modelo **Random Forest** clasificador para fijación ocular. |
| `train_model.py` | Script de experimentación y evaluación comparativa entre múltiples algoritmos de clasificación (Random Forest, SVM, Regresión Logística). |
| `.gitignore` | Configuración para excluir archivos binarios pesados (`.npy`, `.pkl`, carpetas de entorno virtual). |

---

## Flujo de Trabajo y Entrenamiento (`train_gaze.py`)

1. **Tratamiento del Desbalanceo (SMOTE):**
   El dataset original contaba con un sesgo clínico (2,836 casos de Riesgo vs. 284 Típicos). El script aplica **SMOTE** (*Synthetic Minority Over-sampling Technique*) sobre el set de entrenamiento, igualando la distribución a **2,269 muestras por clase**.

2. **Modelo de Clasificación:**
   Se utiliza **Random Forest Classifier** (`n_estimators=100`), aprovechando su arquitectura de ensamble basada en votación mayoritaria por comité de árboles de decisión.

3. **Exportación a Producción:**
   El modelo final entrenado se congela y exporta como un binario en `mejor_modelo_mirada.pkl` para ser consumido por la API backend de la aplicación.

---

## Rendimiento Clínico del Modelo

Evaluación realizada sobre el conjunto de validación independiente (624 muestras reales):

| Métrica | Valor Decimal | Porcentaje |
| :--- | :---: | :---: |
| **Exactitud (Accuracy)** | `0.8830` | **88.3%** |
| **Precisión (Precision)** | `0.9411` | **94.1%** |
| **Sensibilidad (Recall)** | `0.9295` | **92.9%** |
| **F1-Score (Balance Clínico)** | `0.9352` | **93.5%** |
| **Área Bajo la Curva (AUC-ROC)** | `0.8072` | **80.7%** |

---

## Requisitos e Instalación

### Dependencias necesarias:
```bash
pip install numpy scikit-learn imbalanced-learn joblib
