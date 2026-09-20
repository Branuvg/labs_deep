# Registro de uso de IA

## 1. Registro de Prompts por Task

### Task 1: Ingeniería de datos y creación de secuencias
*   **Prompt:** "necesito procesar el dataset de paysim tengo transacciones individuales y quiero agruparlas por remitente para crear secuencias temporales cual es la forma mas eficiente de estructurar estos datos en un dataframe de pandas para luego pasarlos a un modelo de deep learning considerando que debo manejar secuencias de longitud variable y normalizar los montos"
*   **Justificación:** Nos permitió obtener un enfoque funcional para el manejo de datos (uso de `groupby`, `agg`, y `padding`), asegurando que la estructura fuera compatible con PyTorch sin que la IA nos diera una solución de "caja negra" que no entendiéramos.

### Task 2: Diseño del Autoencoder (Etapa A)
*   **Prompt:** "para detectar anomalias en remesas quiero implementar un autoencoder que aprenda a reconstruir secuencias normales como puedo diseñar una arquitectura en pytorch que maneje secuencias de longitud variable y use mecanismos de atencion para que el modelo aprenda a reconstruir solo las transacciones normales"
*   **Justificación:** La respuesta ayudó a definir la arquitectura base (LSTM con atención). Nos obligó a razonar sobre por qué el error de reconstrucción es una buena métrica de anomalía, alineándose con el objetivo del proyecto.

### Task 3: Implementación del bucle de entrenamiento (Etapa A)
*   **Prompt:** "como implemento un bucle de entrenamiento en pytorch para un autoencoder que solo se entrena con casos normales que metrica deberia monitorear en el conjunto de validacion para asegurar que el modelo no este simplemente memorizando los datos"
*   **Justificación:** Fue crucial para entender que debíamos filtrar los datos de entrenamiento y no solo centrarnos en el `loss` de reconstrucción, sino en la capacidad de generalización del modelo.

### Task 4: Estrategia de Transfer Learning (Etapa B)
*   **Prompt:** "tengo el autoencoder de la etapa a entrenado ahora quiero clasificar casos de lavado de dinero etapa b que estrategias de transfer learning me recomiendas para aprovechar las representaciones aprendidas por el autoencoder sin reentrenar todo desde cero"
*   **Justificación:** Nos ayudó a elegir congelar las capas del codificador y añadir una capa de clasificación, lo cual justificamos técnicamente en el reporte como una forma de aprovechar el conocimiento de la "normalidad" previo.

### Task 5: Manejo del desbalance de clases
*   **Prompt:** "mi dataset para la etapa b esta extremadamente desbalanceado muy pocos casos de lavado que funciones de perdida o tecnicas de muestreo me recomiendas en pytorch para que el modelo no ignore la clase minoritaria"
*   **Justificación:** Nos orientó hacia el uso de `WeightedRandomSampler` y `BCEWithLogitsLoss` con `pos_weight`, decisiones que luego explicamos en el reporte como clave para obtener métricas balanceadas.

### Task 6: Experimento de ablación
*   **Prompt:** "necesito demostrar que el diseño de dos etapas a + b es mejor que un clasificador supervisado puro como puedo estructurar un experimento de ablacion para comparar ambos enfoques de manera justa"
*   **Justificación:** Nos ayudó a entender la importancia de fijar las mismas semillas aleatorias y métricas (F1-score, Precision-Recall AUC) para que la comparación fuera metodológicamente sólida.

### Task 7: Análisis de interpretabilidad (Atención)
*   **Prompt:** "como puedo extraer y visualizar los pesos de atencion de una capa de atencion de pytorch para entender que transacciones de una secuencia fueron mas importantes para la prediccion final"
*   **Justificación:** Esencial para cumplir con el componente de interpretabilidad. La IA nos ayudó a mapear los pesos `alpha` a la secuencia original para crear los mapas de calor.

### Task 8: Uso de IA en la redacción del reporte
*   **Prompt:** "tengo este txt con algunos prompts utilizados durante el proyecto, necesito que me ayudes a redactar la seccion de resultados y discusion del reporte de la parte de uso de ia"
*   **Justificación:** La IA nos ayudó a estructurar la sección de Uso de IA, proporcionando ejemplos de lenguaje técnico y sugerencias para mejorar la claridad y coherencia del texto.

## 2. Resumen Ejecutivo
El equipo utilizó herramientas de IA generativa (Claude, Codex y Gemini) como un "programador en pareja" para acelerar tareas de implementación técnica y depuración de código. Las decisiones críticas, como la selección del subconjunto de datos, la justificación de los umbrales de anomalía, y la interpretación de los patrones de lavado detectados, fueron tomadas enteramente por el equipo basándose en los conceptos vistos en clase. La IA facilitó la estructura técnica y la generación del boilerplate, pero el análisis de interpretabilidad y la lógica de negocio en la generación de explicaciones fueron redactados y validados por nosotros para garantizar que el sistema cumpla con un estándar regulatorio real.
