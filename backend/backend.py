import face_recognition
import heapq
import numpy as np
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from rtree import index
import faiss
import random

carpeta_entrada = "../../fotos"  # Ruta de la carpeta que contiene las carpetas con las fotos
carpeta_salida = "../vectores_prueba"  # Ruta de la carpeta donde se guardarán los archivos de texto con los vectores
carpeta_salida_seq = "../vectores_12800"

#Sequential
def knn_sequential(query_vector, k, vectors_folder):
    pq = []
    start_time = perf_counter()
    def calculate_distance(file_path, query_vector):
        try:
            # Cargar el vector de codificación desde el archivo
            vector = np.loadtxt(file_path, delimiter=',')
            
            # Calcular la distancia entre el objeto de consulta y el vector
            query_vector_np = np.array(query_vector)
            vector_np = np.array(vector)
            distance = face_recognition.face_distance([vector_np], query_vector_np)[0]
            
            # Agregar la distancia y el vector a la cola de prioridad negando la distancia
            heapq.heappush(pq, (-distance, file_path))
            
            # Mantener solo los K vecinos más cercanos
            if len(pq) > k:
                heapq.heappop(pq)
        
        except Exception as e:
            print(f"Error al procesar el archivo {file_path}: {str(e)}")
    
    # Obtener la lista de archivos en la carpeta de vectores
    files = [os.path.join(vectors_folder, file_name) for file_name in os.listdir(vectors_folder)]
    
    # Calcular las distancias
    for file_path in files:
        calculate_distance(file_path, query_vector)
    
    # Ordenar los vecinos por distancia de mayor a menor utilizando la función sorted
    neighbors = sorted(pq, key=lambda x: x[0], reverse=False)
    end_time = perf_counter()
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución para knn_sequential: {execution_time} segundos")
    
    return neighbors

#Range
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

#KNN-Rtree
nombres = []

def KNN_Ktree(query_vector, k):
    res = []
    p = index.Property()
    p.dimension = 128  # Dimensión de los vectores
    p.buffering_capacity = 4  # Capacidad de almacenamiento en memoria
    idx = index.Index(properties=p)
    cont = 0
    for file_name in os.listdir(carpeta_salida):
        file_path = os.path.join(carpeta_salida, file_name)
        vector = []
        with open(file_path, "r") as archivo:
            nombres.append(file_name)
            for linea in archivo:
                valor = float(linea.strip())
                vector.append(valor)
        idx.insert(id=cont, coordinates=tuple(vector))
        cont = cont + 1
    start_time = perf_counter()
    neighbors = list(idx.nearest(coordinates=query_vector, num_results=k))
    end_time = perf_counter()
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución para knn_tree: {execution_time} segundos")
    for vecino in neighbors:
        res.append(nombres[vecino])
    return res

#Faiss
def KNN_HighD(vector_imagen, k):
    res = []
    d = 128  # Dimensión de los vectores
    index = faiss.IndexFlatL2(d)
    vectors = []
    for file_name in os.listdir(carpeta_salida):
        file_path = os.path.join(carpeta_salida, file_name)
        with open(file_path, "r") as archivo:
            nombres.append(file_name)
            vector = [float(linea.strip()) for linea in archivo]
            vectors.append(vector)
    samples = np.array(vectors, dtype=np.float32)
    index.add(samples)
    start_time = perf_counter()
    distances, indices = index.search(np.expand_dims(vector_imagen, axis=0), k)
    end_time = perf_counter()
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución para knn_highd: {execution_time} segundos")
    for i in range(k):
        index = indices[0][i]
        nombre = nombres[index]
        distance = distances[0][i]
        res.append((nombre, distance))
    return res

# Ejemplo de uso

#imagen = face_recognition.load_image_file("../Angel Tito.jpg")
#codificaciones = face_recognition.face_encodings(imagen)

#k = 4  # Cantidad de objetos a recuperar
#vectors_folder = carpeta_salida  # Ruta de la carpeta donde se encuentran los archivos con los vectores

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