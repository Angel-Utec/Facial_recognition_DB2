import face_recognition
import heapq
import numpy as np
import os
import matplotlib.pyplot as plt
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
import random

carpeta_entrada = "../fotos"  # Ruta de la carpeta que contiene las carpetas con las fotos
carpeta_salida = "./vectores"  # Ruta de la carpeta donde se guardarán los archivos de texto con los vectores

def knn_sequential(query_vector, k, vectors_folder):
    pq = []
    
    for file_name in os.listdir(vectors_folder):
        file_path = os.path.join(vectors_folder, file_name)
        
        try:
            # Cargar el vector de codificación desde el archivo
            vector = np.loadtxt(file_path, delimiter=',')
            
            # Calcular la distancia entre el objeto de consulta y el vector
            distance = np.linalg.norm(vector - query_vector)
            
            # Agregar la distancia y el vector a la cola de prioridad
            heapq.heappush(pq, (distance, file_name))
            
            # Mantener solo los K vecinos más cercanos
            if len(pq) > k:
                heapq.heappop(pq)
        
        except Exception as e:
            print(f"Error al procesar el archivo {file_path}: {str(e)}")
    
    # Ordenar los vecinos por distancia de menor a mayor
    neighbors = sorted(pq, key=lambda x: x[0])
    
    return neighbors
def busqueda_por_rango(query_vector, radio, vectors_folder):
    resultados = []
    for file_name in os.listdir(vectors_folder):
        file_path = os.path.join(vectors_folder, file_name)
        
        try:
            # Cargar el vector de codificación desde el archivo
            vector = np.loadtxt(file_path, delimiter=',')
            
            # Calcular la distancia entre el objeto de consulta y el vector
            distance = np.linalg.norm(vector - query_vector)

            if distance <= radio:
                resultados.append(distance)
        
        except Exception as e:
            print(f"Error al procesar el archivo {file_path}: {str(e)}")
    
    return resultados

# Ejemplo de uso

imagen = face_recognition.load_image_file("./Angel Tito.jpg")
codificaciones = face_recognition.face_encodings(imagen)

k = 4  # Cantidad de objetos a recuperar
vectors_folder = carpeta_salida  # Ruta de la carpeta donde se encuentran los archivos con los vectores

start_time = perf_counter()
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(knn_sequential, codificaciones, k, vectors_folder) for _ in range(10)]
    
    all_neighbors = []
    for future in futures:
        all_neighbors.append(future.result())
end_time = perf_counter()

execution_time = end_time - start_time
print(f"Tiempo de ejecución para knn_sequential: {execution_time} segundos")

# Imprimir los vecinos encontrados
for neighbors in all_neighbors:
    for distance, file_name in neighbors:
        print(f"Archivo: {file_name}, Distancia: {distance}")


radios = []

for _ in range(3):
    numero = round(random.uniform(0, 2), 2)
    radios.append(numero)

for radio in radios:
    start_time = perf_counter()
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(busqueda_por_rango, codificaciones, radio, vectors_folder) for _ in range(10)]
        
        distancias = []
        for future in futures:
            distancias.extend(future.result())
    end_time = perf_counter()

    execution_time = end_time - start_time
    print(f"Tiempo de ejecución de busqueda por rango: {execution_time} segundos")

    if(len(distancias) == 0):
        print("No se encontro resultados")
    else:
        print(f"Radio: {radio}")
        print(f"Número de resultados: {len(distancias)}")
        print(f"Mínimo: {min(distancias)}")
        print(f"Máximo: {max(distancias)}")
        print(f"Promedio: {np.mean(distancias)}")
        print(f"Desviación estándar: {np.std(distancias)}")
        print()
    
        # Visualización de la distribución de la distancia
        plt.hist(distancias, bins=10)
        plt.xlabel('Distancia')
        plt.ylabel('Frecuencia')
        plt.title(f'Distribución de la distancia (Radio: {radio})')
        plt.show()
