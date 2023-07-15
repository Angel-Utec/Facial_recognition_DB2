import face_recognition
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

carpeta_entrada = "../fotos"  # Ruta de la carpeta que contiene las carpetas con las fotos
carpeta_salida = "./vectores_prueba"  # Ruta de la carpeta donde se guardarán los archivos de texto con los vectores

def procesar_imagen(ruta_imagen, carpeta_salida):
    try:
        # Carga la imagen y obtiene los vectores de codificación
        imagen = face_recognition.load_image_file(ruta_imagen)
        codificaciones = face_recognition.face_encodings(imagen)

        if len(codificaciones) > 0:
            # Selecciona la primera codificación
            codificacion = codificaciones[0]

            # Guarda el vector de codificación en un archivo de texto
            nombre_archivo = f"{os.path.basename(ruta_imagen)[:-4]}.txt"
            ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
            np.savetxt(ruta_salida, codificacion, delimiter=',', fmt='%.20f')
    except Exception as e:
        print(f"Error al procesar la imagen {ruta_imagen}: {str(e)}")

# Obtener la lista de todas las rutas de las imágenes
rutas_imagenes = []
for raiz, carpetas, archivos in os.walk(carpeta_entrada):
    for archivo in archivos:
        ruta_imagen = os.path.join(raiz, archivo)
        rutas_imagenes.append(ruta_imagen)
# Procesar las imágenes en paralelo
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(procesar_imagen, ruta_imagen, carpeta_salida) for ruta_imagen in rutas_imagenes]

# Esperar a que todas las tareas se completen
for future in futures:
    future.result()