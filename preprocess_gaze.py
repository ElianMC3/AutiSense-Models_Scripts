import json
import os
import numpy as np

def extract_gaze_features(gaze_data):
    """Calcula métricas estadísticas descriptivas del comportamiento de la mirada."""
    # Convertir a arreglos de numpy ignorando los nulos en los cálculos estadísticos
    rx = np.array(gaze_data.get('rx', []), dtype=float)
    ry = np.array(gaze_data.get('ry', []), dtype=float)
    rz = np.array(gaze_data.get('rz', []), dtype=float)
    
    # Si los vectores están vacíos o corruptos, descartamos la sesión
    if len(rx) == 0 or np.all(np.isnan(rx)):
        return None

    # Función auxiliar para extraer estadísticas ignorando NaNs (valores perdidos por parpadeos)
    def get_stats(axis_data):
        valid_data = axis_data[~np.isnan(axis_data)]
        if len(valid_data) == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        return [
            np.mean(valid_data),              # Promedio de posición visual
            np.var(valid_data),               # Varianza (Qué tanto se dispersa)
            np.std(valid_data),               # Desviación estándar
            np.percentile(valid_data, 25),    # Cuartil 1
            np.percentile(valid_data, 75)     # Cuartil 3
        ]

    stats_x = get_stats(rx)
    stats_y = get_stats(ry)
    stats_z = get_stats(rz)
    
    # Fusionamos las 15 características estadísticas descriptivas de esta sesión
    return stats_x + stats_y + stats_z

def process_gaze_file(json_path):
    """Abre un JSON y extrae la etiqueta diagnóstica y las métricas de mirada."""
    with open(json_path, 'r') as f:
        try:
            session_data = json.load(f)
        except json.JSONDecodeError:
            return None, None

    # Extracción de etiquetas basada en el esquema clínico del ADOS
    ados_block = session_data.get('ados', {})
    label = 0
    
    if 'preTest' in ados_block:
        # En la mirada, podemos usar 'communication' o 'interaction' como mejor indicador del ADOS,
        # o mantener 'stereotype' para ver correlación. Usemos la afectación de interacción social/comunicación:
        communication_score = ados_block['preTest'].get('communication', 0)
        # Umbral clínico: Puntuación de comunicación mayor a 2 indica riesgo en el espectro
        label = 1 if communication_score > 2 else 0
    elif 'initial' in ados_block:
        label = 1 if ados_block.get('initial', 0) >= 4 else 0
    else:
        return None, None

    # Extraer el bloque eye_gaze
    gaze_data = session_data.get('eye_gaze')
    if not gaze_data:
        return None, None
        
    features = extract_gaze_features(gaze_data)
    if features is None:
        return None, None

    nombre_carpeta_padre = os.path.basename(os.path.dirname(json_path))
    print(f" -> [Mirada] {nombre_carpeta_padre}/{os.path.basename(json_path)} procesado exitosamente.")
    return features, label

# =============================================================
# EJECUCIÓN DEL ESCANEO RECURSIVO PARA EL ALGORITMO 2
# =============================================================
if __name__ == "__main__":
    carpeta_dataset = "dataset"
    X_gaze_list = []
    y_gaze_list = []

    print("=== INICIANDO EXTRACCIÓN DE CARACTERÍSTICAS DE MIRADA (ALGORITMO 2) ===")
    
    if not os.path.exists(carpeta_dataset):
        print(f"Error: No existe la carpeta '{carpeta_dataset}'")
    else:
        for raiz, carpetas, archivos in os.walk(carpeta_dataset):
            for archivo in archivos:
                if archivo.endswith(".json"):
                    ruta_completa = os.path.join(raiz, archivo)
                    features, label = process_gaze_file(ruta_completa)
                    
                    if features is not None:
                        X_gaze_list.append(features)
                        y_gaze_list.append(label)

        if len(X_gaze_list) > 0:
            X_gaze_final = np.array(X_gaze_list)
            y_gaze_final = np.array(y_gaze_list)

            print("\n=== COLECTA DE MIRADA COMPLETA ===")
            print(f"Forma final de X_gaze: {X_gaze_final.shape} (Sesiones, Características Estadísticas)")
            print(f"Forma final de y_gaze: {y_gaze_final.shape}")
            print(f"Distribución: {np.bincount(y_gaze_final)} (0 = Típico, 1 = Riesgo de Comunicación Visual)")

            # Guardamos con nombres únicos para no sobreescribir lo de tus manos
            np.save("X_gaze.npy", X_gaze_final)
            np.save("y_gaze.npy", y_gaze_final)
            print("¡Éxito! Matrices de mirada almacenadas en la raíz.")
        else:
            print("\nNo se encontraron datos de mirada válidos.")