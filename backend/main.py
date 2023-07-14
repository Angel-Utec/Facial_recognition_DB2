import face_recognition
import heapq
import numpy as np
import os
import matplotlib.pyplot as plt
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
import random

carpeta_entrada = "/home/salvador/Documents/BD II/modulo2/proyecto 3/lfw"  # Ruta de la carpeta que contiene las carpetas con las fotos
carpeta_salida = "/home/salvador/Documents/BD II/modulo2/proyecto 3/vertices"  # Ruta de la carpeta donde se guardarán los archivos de texto con los vectores

for raiz, carpetas, archivos in os.walk(carpeta_entrada):
    for archivo in archivos:
        # Crea la ruta completa del archivo de imagen
        ruta_imagen = os.path.join(raiz, archivo)
        try:
            # Carga la imagen y obtiene los vectores de codificación
            imagen = face_recognition.load_image_file(ruta_imagen)
            codificaciones = face_recognition.face_encodings(imagen)
            codificacioneslol = str(tuple(codificaciones[0]))
            
            print(codificacioneslol)
            if len(codificacioneslol) > 0:
                # Guarda el vector de codificación en un archivo de texto
                nombre_archivo = f"{archivo[:-4]}.txt"
                ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
                with open(ruta_salida, 'w') as archivo_salida:
                    archivo_salida.write(codificacioneslol)
        except Exception as e:
            print(f"Error al procesar la imagen {ruta_imagen}: {str(e)}")


