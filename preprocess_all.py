import json
import os
import numpy as np

def clean_and_convert(joint_data):
    """Limpia e interpola los frames perdidos por la cámara."""
    x = np.array(joint_data['x'], dtype=float)
    y = np.array(joint_data['y'], dtype=float)
    z = np.array(joint_data['z'], dtype=float)
    
    matrix = np.array([x, y, z])
    nans = np.isnan(matrix)
    
    if np.any(nans):
        for i in range(3):
            axis = matrix[i]
            valid = ~np.isnan(axis)
            if not np.any(valid):
                axis[:] = 0.0
            else:
                axis[nans[i]] = np.interp(np.flatnonzero(nans[i]), np.flatnonzero(valid), axis[valid])
            matrix[i] = axis
    return matrix.T

def process_single_file(json_path, window_size=250, step_size=125):
    """Procesa un solo archivo JSON soportando esquemas v1.0, v1.1 y v1.2."""
    with open(json_path, 'r') as f:
        try:
            session_data = json.load(f)
        except json.JSONDecodeError:
            print(f" -> Error: El archivo {os.path.basename(json_path)} está corrupto o vacío. Saltando...")
            return None, None
    
    # --- EXTRACCIÓN BLINDADA BASADA EN LAS ESPECIFICACIONES ---
    ados_block = session_data.get('ados', {})
    label = 0
    version_detectada = ""
    
    if 'preTest' in ados_block:
        # Versiones 1.1 y 1.2 (Buscamos la puntuación de estereotipia explícita)
        ados_stereotype = ados_block['preTest'].get('stereotype', 0)
        label = 1 if ados_stereotype > 0 else 0
        version_detectada = "v1.1/v1.2"
    elif 'initial' in ados_block:
        # Versión 1.0 (Usamos el score general inicial con un umbral clínico de corte >= 4)
        ados_initial = ados_block.get('initial', 0)
        label = 1 if ados_initial >= 4 else 0
        version_detectada = "v1.0 (Legacy)"
    else:
        # Archivo sin datos ADOS válidos, lo saltamos para no ensuciar el entrenamiento
        return None, None

    # El procesamiento del esqueleto
    skeleton = session_data.get('skeleton', {})
    try:
        sholder_c = clean_and_convert(skeleton['sholder_center'])
        hand_r    = clean_and_convert(skeleton['hand_right'])
        hand_l    = clean_and_convert(skeleton['hand_left'])
        wrist_r   = clean_and_convert(skeleton['wrist_right'])
        wrist_l   = clean_and_convert(skeleton['wrist_left'])
    except KeyError:
        # Si a un JSON le falta alguna articulación requerida, evitamos el crash
        return None, None

    # -------------------------------------------------------------
    # NUEVO PROTECTOR DE ARCHIVOS VACÍOS O MUY CORTOS
    # -------------------------------------------------------------
    n_frames = hand_r.shape[0]
    
    if n_frames < window_size:
        nombre_carpeta_padre = os.path.basename(os.path.dirname(json_path))
        print(f" -> [Saltado] {nombre_carpeta_padre}/{os.path.basename(json_path)}: Archivo vacío o muy corto ({n_frames} frames).")
        return None, None
    # -------------------------------------------------------------

    # Normalización Relativa y Velocidades (El resto del código sigue exactamente igual...)
    hand_r_rel, hand_l_rel = hand_r - sholder_c, hand_l - sholder_c
    wrist_r_rel, wrist_l_rel = wrist_r - sholder_c, wrist_l - sholder_c

    vel_hand_r  = np.diff(hand_r_rel, axis=0, prepend=hand_r_rel[[0]])
    vel_hand_l  = np.diff(hand_l_rel, axis=0, prepend=hand_l_rel[[0]])
    vel_wrist_r = np.diff(wrist_r_rel, axis=0, prepend=wrist_r_rel[[0]])
    vel_wrist_l = np.diff(wrist_l_rel, axis=0, prepend=wrist_l_rel[[0]])

    features_matrix = np.hstack([
        hand_r_rel, hand_l_rel, wrist_r_rel, wrist_l_rel,
        vel_hand_r, vel_hand_l, vel_wrist_r, vel_wrist_l
    ]) 

    windows, labels = [], []
    start_frame = 0
    while start_frame + window_size <= n_frames:
        windows.append(features_matrix[start_frame:start_frame + window_size, :])
        labels.append(label)
        start_frame += step_size 

    # Extraemos el nombre de la carpeta padre (ej. User 3) para mostrar un log claro
    nombre_carpeta_padre = os.path.basename(os.path.dirname(json_path))
    print(f" -> [{version_detectada}] {nombre_carpeta_padre}/{os.path.basename(json_path)}: {n_frames} frames -> {len(windows)} ventanas (Clase {label})")
    return windows, labels

# =============================================================
# SCRIPT PRINCIPAL: ESCANEO EN SUB-CARPETAS (RECURSIVO)
# =============================================================
if __name__ == "__main__":
    carpeta_dataset = "dataset"
    all_windows = []
    all_labels = []

    print("=== INICIANDO PROCESAMIENTO RECURSIVO EN LOTE (OPCIÓN 2) ===")
    
    if not os.path.exists(carpeta_dataset):
        print(f"Error: No existe la carpeta '{carpeta_dataset}' en la raíz del proyecto.")
    else:
        # os.walk recorre de forma profunda subcarpetas, carpetas de usuarios y archivos
        for raiz, carpetas, archivos in os.walk(carpeta_dataset):
            for archivo in archivos:
                if archivo.endswith(".json"):
                    # Construye la ruta completa sin importar qué tan oculto esté el archivo
                    ruta_completa = os.path.join(raiz, archivo)
                    
                    windows, labels = process_single_file(ruta_completa)
                    
                    if windows is not None and len(windows) > 0:
                        all_windows.extend(windows)
                        all_labels.extend(labels)

        if len(all_windows) > 0:
            X_final = np.array(all_windows)
            y_final = np.array(all_labels)

            print("\n=== COLECTA COMPLETA EXTRACCIÓN RECURSIVA ===")
            print(f"Forma final de X_data: {X_final.shape}")
            print(f"Forma final de y_data: {y_final.shape}")
            
            conteo = np.bincount(y_final)
            cant_0 = conteo[0] if len(conteo) > 0 else 0
            cant_1 = conteo[1] if len(conteo) > 1 else 0
            print(f"Distribución real de clases: {cant_0} Normales (0) vs {cant_1} Estereotipias (1)")

            # Guardar en la raíz listo para train_model.py
            np.save("X_data.npy", X_final)
            np.save("y_data.npy", y_final)
            print("\n¡Éxito! Matrices unificadas guardadas en la raíz.")
        else:
            print("\nNo se pudieron extraer ventanas válidas de ninguna subcarpeta.")   