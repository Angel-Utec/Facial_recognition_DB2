# Proyecto 3 de Base de datos
## Librerías utilizadas
Las tecnicas que se utilizan son:
- El KD-tree es una estructura de datos de árbol binario que organiza puntos en un espacio multidimensional. En cada nivel del árbol, se divide el espacio en dos hiperplanos mediante un hiperplano ortogonal a uno de los ejes. Esto permite un rápido descarte de regiones del espacio que no contienen los vecinos más cercanos buscados.Cuando se utiliza el algoritmo KNN con KD-trees, el árbol se construye utilizando los puntos de entrenamiento como nodos del árbol. Luego, durante la fase de búsqueda, se recorre el árbol de manera eficiente para encontrar los vecinos más cercanos a un punto objetivo.El uso de KD-trees puede reducir significativamente el tiempo de búsqueda de vecinos más cercanos en comparación con un enfoque de fuerza bruta que compara todos los puntos entre sí. Sin embargo, es importante tener en cuenta que la eficiencia del KD-tree puede verse afectada por la dimensionalidad del espacio y la distribución de los puntos.Además del KD-tree, existen otras estructuras de datos que se pueden utilizar para acelerar la búsqueda de vecinos más cercanos en el algoritmo KNN, como los Ball Trees, los Cover Trees y los VP-Trees, entre otros. Cada una de estas estructuras tiene sus propias características y trade-offs en términos de tiempo de construcción del árbol y tiempo de búsqueda, por lo que la elección de la estructura adecuada depende del contexto y los requisitos específicos de la aplicación.
Como se realiza el KNN Search y el Range Search (si es que lo soporta)

- El R-tree se aplican técnicas y estrategias adicionales para mitigar los efectos de la maldición de la dimensionalidad y mejorar la eficiencia de la búsqueda. Algunas de las técnicas comunes utilizadas en KNN-HighD son:

Reducción de dimensionalidad: Se utilizan técnicas de reducción de dimensionalidad, como el Análisis de Componentes Principales (PCA), para proyectar los datos de alta dimensionalidad en un espacio de menor dimensión. Esto ayuda a preservar la estructura y las relaciones entre los datos, al tiempo que reduce la dimensionalidad, lo que puede facilitar la búsqueda de vecinos más cercanos.

Estructuras de índice especiales: Se emplean estructuras de datos especializadas, como KD-trees, Ball Trees, o índices basados en grafos, para organizar y acelerar la búsqueda de vecinos más cercanos en espacios de alta dimensión. Estas estructuras de índice están diseñadas para manejar eficientemente los desafíos de la maldición de la dimensionalidad.

Hashing sensible a la localidad: Se utilizan técnicas de hashing, como Locality Sensitive Hashing (LSH), que asignan puntos similares a cubetas cercanas utilizando funciones de hash sensibles a la localidad. Esto permite agrupar puntos similares y facilita la búsqueda de vecinos más cercanos.

Aprovechamiento de hardware especializado: Se pueden utilizar implementaciones de KNN-HighD que aprovechan la aceleración en hardware especializado, como GPU (Graphics Processing Unit) para mejorar el rendimiento de la búsqueda en espacios de alta dimensión. Por ejemplo, Faiss es una biblioteca que ofrece índices de GPU eficientes para búsqueda de vecinos más cercanos.

KNN-HighD combina estas técnicas y estrategias para mejorar la eficiencia de la búsqueda de vecinos más cercanos en espacios de alta dimensión. La elección de las técnicas y la configuración depende del contexto específico, los requisitos de la aplicación y las características de los datos.
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
Tablas y gráficos de los resultados
Análisis y discusión
