import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split 
from sklearn.utils import class_weight  # <-- CORRECCIÓN 1: Para importar el balanceador matemático
import os

# =============================================================
# 1. CARGAR LAS MATRICES REALES DESDE LA RAÍZ
# =============================================================
if not os.path.exists("X_data.npy") or not os.path.exists("y_data.npy"):
    raise FileNotFoundError("Error: Corre primero 'preprocess_all.py' para generar las matrices.")

X = np.load("X_data.npy")
y = np.load("y_data.npy").reshape(-1, 1) # Asegura forma de columna automáticamente

print(f"=== BASE DE DATOS CLÍNICA CARGADA ===")
print(f" -> Matriz de características X: {X.shape}")
print(f" -> Vector de etiquetas clínicas y: {y.shape}")

# Conteo real de tus clases (14k vs 183k)
conteo_clases = np.bincount(y.flatten())
print(f" -> Distribución: {conteo_clases[0]} Normales (0) vs {conteo_clases[1]} Estereotipias (1)")

# =============================================================
# 2. CORRECCIÓN 2: BALANCEO MATEMÁTICO DE CLASES (Class Weights)
# =============================================================
# Como el 92% de tus datos son Clase 1, esto le enseña a la red que la Clase 0 vale 13 veces más
clases_unicas = np.unique(y)
pesos = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=clases_unicas,
    y=y.flatten()
)
diccionario_pesos = dict(zip(clases_unicas, pesos))
print(f" -> Pesos de compensación clínicos calculados: {diccionario_pesos}")

# =============================================================
# 3. CORRECCIÓN 3: DIVISIÓN STRATIFIED (80% / 20%)
# =============================================================
# Usamos 'stratify=y' para asegurar que el porcentaje de datos normales sea idéntico en Train y Validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================
# 4. DISEÑO DE LA RED NEURONAL RECURRENTE (LSTM)
# =============================================================
model = Sequential([
    # Primera capa LSTM: Procesa los 250 frames temporales con tus 24 variables cinemáticas
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
    BatchNormalization(), 
    Dropout(0.3),         
    
    # Segunda capa LSTM: Consolida los ritmos y frecuencias del movimiento
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    
    # Capa Densa Intermedia
    Dense(16, activation='relu'),
    
    # Capa de Salida (Sigmoid entrega una probabilidad entre 0.0 y 1.0)
    Dense(1, activation='sigmoid')
])

# =============================================================
# 5. COMPILACIÓN DEL MODELO
# =============================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Recall(name='recall'),      # Crucial para detectar todos los riesgos reales
        tf.keras.metrics.Precision(name='precision')  # Crucial para evitar falsas alarmas en la app
    ]
)

model.summary()

# =============================================================
# 6. CORRECCIÓN 4: GUARDADO AUTOMÁTICO INTELIGENTE (Callbacks)
# =============================================================
# En lugar de guardar el modelo a ciegas al final, guardamos solo cuando el modelo mejora en val_loss
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'modelo_deteccion_autismo.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# =============================================================
# 7. ENTRENAMIENTO INTENSIVO OPTIMIZADO
# =============================================================
print("\nIniciando entrenamiento masivo industrial...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,                     # CORRECCIÓN 5: Subimos a 30 épocas para que la LSTM madure
    batch_size=64,                 # CORRECCIÓN 6: Bloques de 64 para exprimir tu RTX 3050 y tus 16GB RAM
    class_weight=diccionario_pesos, # Activamos el balanceo para mitigar el desbalanceo inverso
    callbacks=[checkpoint],        # Salvavidas de guardado óptimo
    verbose=1
)

print("\n¡Éxito total! El mejor modelo optimizado ha quedado guardado en la raíz.")