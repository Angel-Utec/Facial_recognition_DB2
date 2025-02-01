# Proyecto 3 de Base de datos

Link del video de presentacion del proyecto: Video[https://drive.google.com/drive/folders/10LyXZA-tku7_sNY8uhTdAg_TAUUGABAF?usp=sharing]

## Sequential
En este caso no se utilizo ninguna libreria para calcular las distancias para la tecnica del sequential, ya que como su mismo nombre lo dice este busca de forma secuencial cada imagen, pero si se uso la libreria concurrent.futures para poder paralelizar y optimizar la busqueda lo cual nos favorecio bastante al momento de hacer la busqueda.

Este calculaba la distancia entre cada imagen con la query con la funcion "face.distance" de la libreria "face_recognition", para posterior mandar los datos a un Heap, el cual almacena los resultados y el tamaño del heap
es conforme al Top-K pasada por el usuario que quiera realizar la busqueda, haciendo que solo quede los "k" imagenes mas cercanas.

## K-tree
Se utilizo la libreria "rtree" la cual hace uso de indices espaciales, como R-tree, son estructuras de datos que permiten indexar y realizar consultas eficientes sobre objetos espaciales en un espacio multidimensional.

En este caso se itero dentro de los archivos guardados en memoria secundaria para agregar datos al indice,le proporcionamos un identificador único (ID), el cual era un numero del 0 a la cantidad de datos que se estaba utilizando en las diferentes pruebas, y un bounding box (caja delimitadora) que represente la ubicación del objeto en el espacio multidimensional. La caja delimitadora puede ser una lista o tupla que contenga las coordenadas mínimas y máximas del objeto, en este caso contenia el vector de nuestra imagen vectorizada.

Una vez creado el indice solo bastaba con usar la funcion idx.nearest el cual calculaba la distancia entre los parametros que se le otorgaba.

### KD-tree con el metodo de Faiss

Usamos la libreria "Faiss" la cual utiliza un enfoque de construcción de KD-tree llamado "Recursión Mediana". En este enfoque, se selecciona una dimensión (característica) para dividir el espacio en dos mitades, de modo que la mediana de los puntos 
en esa dimensión se convierte en el punto de corte. Luego, los puntos se distribuyen a ambos lados de la mediana y se construyen dos subárboles para cada mitad. Este proceso se repite recursivamente hasta que cada hoja 
del árbol contiene solo un punto o hasta que se alcance un tamaño de hoja predeterminado.
Es una estructura de datos de árbol binario que organiza puntos en un espacio multidimensional. En cada nivel del árbol, se divide el espacio en dos hiperplanos mediante un hiperplano ortogonal a uno de los ejes.

Una vez que el KD-tree está construido, es posible realizar búsquedas eficientes de vecinos más cercanos. Para encontrar los vecinos más cercanos a un determinado vector de consulta, el árbol se recorre utilizando un 
proceso similar a la búsqueda binaria. Comienza en la raíz del árbol y desciende a través de los nodos, eligiendo el subárbol que esté más cerca del vector de consulta en cada paso.

### Librerias extra usadas:

- matplotlib.pyplot: Para el grafico de las busquedas por rango
- time: La cual usamos para hacer el calculo de los tiempos de ejecucion
- face_recognition: Para la vectorizacion de las imagenes y calculo de distancias dentro de la tecnica del sequential

## Imagenes de nuestra aplicacion:

[![image.png](https://i.postimg.cc/Hn84nkrR/image.png)](https://postimg.cc/BPs1wscB)

[![image.png](https://i.postimg.cc/JzLgf9gM/image.png)](https://postimg.cc/6yMYR1tm)

## Análisis de la maldición de la dimensionalidad y como mitigarlo
La maldición de la dimensionalidad es un desafío que surge cuando trabajamos con conjuntos de datos en espacios de alta dimensión. Se refiere a una serie de problemas que se presentan debido al crecimiento exponencial de la dimensionalidad de los datos y que afectan la eficiencia y la calidad de los resultados en diversas tareas de análisis de datos.

Los metodos para mitigarla son las siguientes:
- Esparsidad de los datos: A medida que aumenta la dimensionalidad, los datos tienden a estar dispersos en el espacio. Esto significa que la densidad de los datos disminuye, y encontrar vecinos cercanos se vuelve más difícil. Para mitigar esto, se pueden utilizar técnicas de reducción de dimensionalidad, como PCA, que proyectan los datos en un espacio de menor dimensión y pueden ayudar a concentrar los datos.

- Distancias ambiguas: En espacios de alta dimensión, las distancias entre puntos se vuelven menos significativas y ambiguas. Esto se debe a que la mayoría de los puntos están aproximadamente a la misma distancia entre sí. Una forma de mitigar esto es utilizar medidas de distancia más robustas, como la distancia coseno o la distancia de Mahalanobis, que se ajustan mejor a las características de los datos en espacios de alta dimensión.

- Relevancia de las características: En espacios de alta dimensión, no todas las características son igualmente informativas o relevantes. Algunas características pueden ser redundantes o no aportar mucha información discriminativa. Por lo tanto, es importante realizar una selección o extracción de características adecuada para reducir la dimensionalidad y eliminar características irrelevantes, lo que puede mejorar la calidad de los resultados.

- Reducción de dimensionalidad: Una estrategia común para mitigar la maldición de la dimensionalidad es utilizar técnicas de reducción de dimensionalidad, como PCA, LDA (Análisis Discriminante Lineal) o t-SNE (t-Distributed Stochastic Neighbor Embedding). Estas técnicas permiten proyectar los datos en un espacio de menor dimensión preservando, en la medida de lo posible, la estructura y las relaciones entre los puntos.

- Uso de estructuras de índice especializadas: En lugar de realizar una búsqueda exhaustiva en espacios de alta dimensión, se pueden utilizar estructuras de índice especializadas, como KD-trees, Ball Trees o índices basados en grafos, que están diseñadas para manejar eficientemente los desafíos de la maldición de la dimensionalidad y permiten búsquedas más rápidas de vecinos más cercanos.

- Generación de datos sintéticos: En algunos casos, se pueden utilizar técnicas de generación de datos sintéticos para aumentar la densidad de datos en regiones específicas del espacio. Esto puede ayudar a contrarrestar la esparsidad de los datos y mejorar la eficiencia de la búsqueda.
## Experimentación
### Tabla de tiempos
[![Captura-de-pantalla-2023-07-14-222829.png](https://i.postimg.cc/vBWW9NYH/Captura-de-pantalla-2023-07-14-222829.png)](https://postimg.cc/mzgzWV10)

### Grafico
[![Figure-1.png](https://i.postimg.cc/76v35kyW/Figure-1.png)](https://postimg.cc/R3R6Y2h7)

Viendo y analizando los graficos observamos que el metodo mas eficiente fue el hecho por HighD que utlizaba la libreria "Faiss", siguiendolo el hecho mediante el K-Tree y por ultimo el sequential. Todas suben conforme se 
va aumentando los datos, pero solo en el sequential se puede lograr apreciar cambios visibles en su tiempo de ejecucion, en comparacion a las otras tecnicas las cuales se ve un mayor cambio si solo se toma en cuenta el primer y ultimo punto, ya que alli se podria apreciar un cambio. El grafico se hizo de forma logaritmica para que se pudiera apreciar esos cambios.
