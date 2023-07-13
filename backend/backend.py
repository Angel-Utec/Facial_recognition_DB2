import face_recognition
import heapq
import numpy as np
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import random

carpeta_entrada = "../../fotos"  # Ruta de la carpeta que contiene las carpetas con las fotos
carpeta_salida = "../vectores_12800"  # Ruta de la carpeta donde se guardarán los archivos de texto con los vectores

# def procesar_imagen(ruta_imagen, carpeta_salida):
#     try:
#         # Carga la imagen y obtiene los vectores de codificación
#         imagen = face_recognition.load_image_file(ruta_imagen)
#         codificaciones = face_recognition.face_encodings(imagen)
        
#         if len(codificaciones) > 0:
#             # Guarda el vector de codificación en un archivo de texto
#             nombre_archivo = f"{os.path.basename(ruta_imagen)[:-4]}.txt"
#             ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
#             np.savetxt(ruta_salida, codificaciones[0], delimiter=',')
#     except Exception as e:
#         print(f"Error al procesar la imagen {ruta_imagen}: {str(e)}")

# # Obtener la lista de todas las rutas de las imágenes
# rutas_imagenes = []
# for raiz, carpetas, archivos in os.walk(carpeta_entrada):
#     for archivo in archivos:
#         ruta_imagen = os.path.join(raiz, archivo)
#         rutas_imagenes.append(ruta_imagen)

# # Procesar las imágenes en paralelo
# with ThreadPoolExecutor() as executor:
#     futures = [executor.submit(procesar_imagen, ruta_imagen, carpeta_salida) for ruta_imagen in rutas_imagenes]

# # Esperar a que todas las tareas se completen
# for future in futures:
#     future.result()

# for raiz, carpetas, archivos in os.walk(carpeta_entrada):
#     for archivo in archivos:
#         # Crea la ruta completa del archivo de imagen
#         ruta_imagen = os.path.join(raiz, archivo)
#         try:
#             # Carga la imagen y obtiene los vectores de codificación
#             imagen = face_recognition.load_image_file(ruta_imagen)
#             codificaciones = face_recognition.face_encodings(imagen)
            
#             if len(codificaciones) > 0:
#                 # Guarda el vector de codificación en un archivo de texto
#                 nombre_archivo = f"{archivo[:-4]}.txt"
#                 ruta_salida = os.path.join(carpeta_salida, nombre_archivo)
#                 np.savetxt(ruta_salida, codificaciones[0], delimiter=',')
#         except Exception as e:
#             print(f"Error al procesar la imagen {ruta_imagen}: {str(e)}")

def knn_sequential(query_vector, k, vectors_folder):
    pq = []
    cont = 0
    def calculate_distance(file_path, query_vector):
        try:
            # Cargar el vector de codificación desde el archivo
            vector = np.loadtxt(file_path, delimiter=',')
            
            # Calcular la distancia entre el objeto de consulta y el vector
            query_vector_np = np.array(query_vector)
            vector_np = np.array(vector)
            distance = face_recognition.face_distance([vector_np], query_vector_np)[0]
            
            # Agregar la distancia y el vector a la cola de prioridad
            heapq.heappush(pq, (distance, file_path))
            
            # Mantener solo los K vecinos más cercanos
            if len(pq) >= k:
                heapq.heappop(pq)
        
        except Exception as e:
            print(f"Error al procesar el archivo {file_path}: {str(e)}")
    
    # Obtener la lista de archivos en la carpeta de vectores
    files = [os.path.join(vectors_folder, file_name) for file_name in os.listdir(vectors_folder)]
    
    # Calcular las distancias
    for file_path in files:
        cont = cont + 1
        print(cont)
        calculate_distance(file_path, query_vector)
    
    # Ordenar los vecinos por distancia de menor a mayor
    neighbors = sorted(pq, key=lambda x: x[0])
    
    return neighbors


def busqueda_por_rango(query_vector, radio, vectors_folder):
    distancias = []
    #nombre = []
    
    def check_distance(file_path):
        try:
            # Cargar el vector de codificación desde el archivo
            vector = np.loadtxt(file_path, delimiter=',')
            
            # Calcular la distancia entre el objeto de consulta y el vector
            distance = np.linalg.norm(vector - query_vector)

            if distance <= radio:
                distancias.append(distance)
                #nombre.append(file_path)
        
        except Exception as e:
            print(f"Error al procesar el archivo {file_path}: {str(e)}")
    
    with ThreadPoolExecutor() as executor:
        # Obtener la lista de archivos en la carpeta de vectores
        files = [os.path.join(vectors_folder, file_name) for file_name in os.listdir(vectors_folder)]
        
        # Verificar las distancias en paralelo
        executor.map(check_distance, files)
    #for x in nombre:
    #    print(f"Coincidencia con: {x}")
    return distancias

# Ejemplo de uso

# imagen = face_recognition.load_image_file("../Angel Tito.jpg")
# codificaciones = face_recognition.face_encodings(imagen)

# k = 4  # Cantidad de objetos a recuperar
# vectors_folder = carpeta_salida  # Ruta de la carpeta donde se encuentran los archivos con los vectores

# start_time = perf_counter()
# neighbors = knn_sequential(codificaciones, k, vectors_folder)
# end_time = perf_counter()

# execution_time = end_time - start_time
# print(f"Tiempo de ejecución para knn_sequential: {execution_time} segundos")
# # Imprimir los vecinos encontrados
# for distance, file_name in neighbors:
#     print(f"Archivo: {file_name}, Distancia: {distance}")

# radios = []

# for _ in range(3):
#     numero = round(random.uniform(0, 2), 2)
#     radios.append(numero)

# for radio in radios:
#     start_time = perf_counter()
#     distancias = busqueda_por_rango(codificaciones, radio, vectors_folder)
#     end_time = perf_counter()

#     execution_time = end_time - start_time
#     print(f"Tiempo de ejecución de busqueda por rango: {execution_time} segundos")

#     if(len(distancias) == 0):
#         print(f"Radio: {radio}")
#         print("No se encontro resultados")
#     else:
#         print(f"Radio: {radio}")
#         print(f"Número de resultados: {len(distancias)}")
#         print(f"Mínimo: {min(distancias)}")
#         print(f"Máximo: {max(distancias)}")
#         print(f"Promedio: {np.mean(distancias)}")
#         print(f"Desviación estándar: {np.std(distancias)}")
#         print()
    
#         # Visualización de la distribución de la distancia
#         plt.hist(distancias, bins=10)
#         plt.xlabel('Distancia')
#         plt.ylabel('Frecuencia')
#         plt.title(f'Distribución de la distancia (Radio: {radio})')
#         plt.show()