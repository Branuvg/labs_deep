# CC3045 – Deep Learning
## Laboratorio 10

## Instrucciones

En este laboratorio usted explica las decisiones de dos modelos con las técnicas vistas en clase. En visión, implementa Grad-CAM desde cero sobre un ResNet-18 y verifica numéricamente su relación con CAM. En datos tabulares, trabaja como parte del equipo de riesgo de modelo de una entidad financiera que debe justificar decisiones de crédito: calcula valores SHAP, implementa LIME desde cero y construye explicaciones contrafactuales. Finalmente pone a prueba si esas explicaciones merecen confianza.

Cada respuesta de análisis debe apoyarse en los números y las figuras que usted produjo. Una respuesta que podría escribirse sin ejecutar el código no recibirá puntos.

Además de PyTorch, torchvision, scikit-learn, NumPy, SciPy y Matplotlib, que ya vienen disponibles en Google Colab, necesita instalar la librería `shap`:

```
pip install shap
```

> **NOTA:** Todo el código debe estar escrito con PyTorch, NumPy y scikit-learn. En los Tasks 1 y 4 no se permite `pytorch-grad-cam`, `captum` ni librerías similares. En el Task 2 puede usar la librería `shap` y sus gráficos, y el cálculo con la fórmula de Shapley de la Task 2.5 debe estar escrito por usted. En el Task 3 no se permite la librería `lime`. Fije una semilla aleatoria y repórtela.

---

## Task 1 (Entrega Parcial)

### Task 1.1

Cargue con torchvision el modelo ResNet-18 con los pesos preentrenados de ImageNet y póngalo en modo de evaluación. Preprocese cada imagen con redimensión a 256 píxeles en el lado corto, recorte central de 224 por 224 y normalización por canal con la media (0.485, 0.456, 0.406) y la desviación estándar (0.229, 0.224, 0.225) de ImageNet.

Elija tres imágenes propias (de internet o de su cámara, indicando la fuente) con estas características:

- (a) un objeto claro y centrado de una clase de ImageNet;
- (b) dos objetos de clases distintas de ImageNet, por ejemplo un gato y un perro;
- (c) un objeto sobre un fondo que pueda engañar al modelo, por ejemplo un objeto fuera de su contexto habitual.

Para cada imagen reporte las cinco clases de mayor probabilidad con sus probabilidades. Para la imagen (c), defina además el recuadro que encierra al objeto y reporte sus coordenadas en la imagen de 224 por 224.

### Task 1.2

Implemente Grad-CAM desde cero sobre la última capa convolucional de la red (`layer4`). Registre un hook de avance que capture las activaciones $A$ (512 canales de $7 \times 7$) y un hook de retroceso que capture $\partial y^c / \partial A$, respetando estas reglas:

- Retropropague desde el logit $y^c$ de la clase, antes del softmax, no desde la probabilidad ni desde una pérdida.
- Calcule $\alpha_k^c$ como el promedio espacial del gradiente de cada canal $k$.
- Forme $L^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$, amplíelo a $224 \times 224$ con interpolación bilineal y normalícelo con su mínimo y su máximo.

Genere el mapa de la clase de mayor probabilidad de cada imagen y superpóngalo a la imagen. Para la imagen (b) genere además el mapa de la clase del segundo objeto.

### Task 1.3

Aproveche que el ResNet-18 termina en pooling global y una capa lineal para calcular CAM directamente desde los pesos de esa capa final. Realice:

a. Calcule $M_{ij}^c = \sum_k w_k^c A_{ij}^k$ con los pesos de la capa `fc` para la clase de cada imagen, sin usar gradientes.

b. Verifique la descomposición del puntaje: $\frac{1}{Z}\sum_{i,j} M_{ij}^c + b_c$ debe igualar el logit $y^c$, con $Z = 49$. Reporte ambos valores y el error absoluto para cada imagen.

c. Verifique la relación con Grad-CAM: calcule $L^c$ sin ReLU y sin normalización, multiplíquelo por $Z$ y compárelo con $M^c$. Reporte el error absoluto máximo sobre las 49 posiciones para cada imagen.

d. Aplique ReLU y normalización mínimo-máximo a ambos mapas y reporte el error absoluto máximo entre ellos.

### Task 1.4

Considere y conteste con base en sus propios resultados:

a. Obtenga numéricamente $\alpha_k^c$ para la clase de la imagen (a) y compárelo con $w_k^c / Z$ para los cinco canales de mayor peso. Reporte los valores y el error relativo. Explique en qué paso de la derivación vista en clase se usa que la derivada de $y^c$ respecto a $A_{ij}^k$ es la misma en todas las posiciones, y por qué los errores que observa son del orden de la precisión de punto flotante y no exactamente cero.

b. Para la imagen (b), compare los mapas de las dos clases: tome, en cada mapa, el conjunto de píxeles del 20% de mayor valor y reporte la intersección sobre la unión (IoU) entre ambos conjuntos. ¿El mapa discrimina entre las dos clases? Considerando que cada posición del mapa original cubre $32 \times 32$ píxeles de la entrada, indique qué límite impone eso a la localización de objetos pequeños o cercanos entre sí.

c. Para la imagen (c), calcule la fracción de la masa del mapa normalizado que cae dentro del recuadro del objeto. Luego haga una prueba causal: reemplace por gris todo lo que queda fuera del recuadro y mida el cambio en la probabilidad de la clase. ¿Coinciden el mapa y la prueba causal en si el modelo usa el fondo? Reporte los números y concluya si hay evidencia de un atajo.

---

## Task 2 (Entrega Parcial)

### Task 2.1

Usted trabaja en el equipo de riesgo de modelo de una entidad financiera que debe explicar sus decisiones de crédito. Descargue el conjunto German Credit, con 1 000 solicitudes y 20 variables (7 numéricas y 13 categóricas), con la siguiente instrucción:

```python
from sklearn.datasets import fetch_openml
datos = fetch_openml("credit-g", version=1, as_frame=True)
```

Codifique la clase `good` como 1 y `bad` como 0, y las variables categóricas con un codificador ordinal. Los códigos son arbitrarios: trátelos como numéricos y deje constancia de esa limitación. Divida el conjunto en 80% de entrenamiento y 20% de prueba, de forma estratificada, y entrene un clasificador de gradient boosting de scikit-learn con los valores de esta tabla:

| Parámetro | Valor |
|---|---|
| Semilla aleatoria | 42 |
| Número de árboles | 200 |
| Profundidad máxima de cada árbol | 3 |
| Tamaño del fondo para SHAP | 100 instancias |

Reporte la exactitud y el AUC sobre el conjunto de prueba.

### Task 2.2

Calcule los valores SHAP del conjunto de prueba con el explicador de árboles de la librería `shap`, en su versión intervencional, sobre la salida del modelo en log-odds y con un fondo de 100 instancias de entrenamiento elegidas al azar. Realice:

a. Verifique la eficiencia: para cada instancia de prueba, el valor base más la suma de los $\phi_j$ debe igualar la salida del modelo en log-odds. Reporte el error absoluto máximo.

b. Verifique que el valor base coincide con el promedio de la salida del modelo en log-odds sobre el fondo, y reporte ambos números.

c. Calcule la importancia global $I_j = \frac{1}{N}\sum_{n=1}^{N}\left|\phi_j^{(n)}\right|$ y la importancia por permutación de scikit-learn sobre el conjunto de prueba (métrica AUC, 10 repeticiones). Reporte las cinco variables principales de cada una y la correlación de Spearman entre los dos rankings completos.

d. Elija un cliente de prueba con probabilidad de `good` mayor que 0.8 y otro con probabilidad menor que 0.2. Dibuje la cascada (waterfall) de cada uno y describa con palabras sus tres contribuciones más grandes. Incluya también el gráfico beeswarm global.

### Task 2.3

Para estudiar qué hace SHAP con variables redundantes, elija la variable numérica de mayor $I_j$ del Task 2.2 y cree una copia con ruido, $x_{dup} = x + 0.05\,\sigma_x\,\epsilon$, donde $\sigma_x$ es la desviación estándar de la variable y $\epsilon \sim \mathcal{N}(0,1)$. Reentrene el modelo con 21 variables y la misma semilla, y calcule los valores SHAP con dos configuraciones: la intervencional con fondo, y la dependiente de la trayectoria del árbol (`tree_path_dependent`), que no usa fondo, aprovecha la estructura de los árboles y aproxima la versión condicional. Presente una tabla con el promedio de $|\phi|$ de la variable original y de su copia en cada configuración, y el promedio de $|\phi_{orig} + \phi_{copia}|$. Compárela con el $I_j$ de la variable original en el modelo sin copia.

### Task 2.4

Compare KernelSHAP con TreeSHAP. Realice:

a. Calcule con el explicador KernelSHAP de la librería `shap`, sobre la salida del modelo en log-odds y con 2 000 muestras por instancia, los valores SHAP de 10 instancias de prueba, para fondos de 10, 50 y 100 instancias.

b. Para cada fondo calcule el error absoluto medio respecto a TreeSHAP intervencional calculado con ese mismo fondo, y mida el tiempo de cómputo.

### Task 2.5

Verifique el cálculo con la fórmula de Shapley escrita por usted, sin usar `shap` en la parte de cálculo. Entrene un segundo modelo con las mismas condiciones pero solo con seis variables numéricas (`duration`, `credit_amount`, `installment_commitment`, `residence_since`, `age`, `existing_credits`) y utilice un fondo de 50 instancias. Para 5 instancias de prueba:

a. Defina $v(S)$ como el promedio de la salida del modelo en log-odds sobre el fondo, fijando en la instancia los valores de las variables de $S$ y tomando del fondo los de las demás.

b. Enumere las $2^6 = 64$ coaliciones y calcule cada $\phi_i$ con la fórmula vista en clase, con los pesos con factoriales.

c. Compare con el explicador de árboles intervencional y reporte el error máximo. Verifique también la eficiencia de sus $\phi_i$, es decir, que sumen $v(N) - v(\emptyset)$.

### Task 2.6

Considere y conteste con base en sus propios resultados:

a. Tome el cliente rechazado del Task 2.2. Calcule $p(x) = \sigma(f(x))$, la probabilidad base $\sigma(\phi_0)$ y, para cada variable $j$, el efecto individual en probabilidad $\sigma(\phi_0 + \phi_j) - \sigma(\phi_0)$. Compare la suma de esos efectos con $p(x) - \sigma(\phi_0)$ y reporte ambos números. Explique por qué los valores SHAP suman exactamente en log-odds pero no en probabilidad.

b. Identifique la variable con mayor diferencia de posición entre su ranking de $I_j$ y el de permutación. Proponga una hipótesis respaldada por datos (por ejemplo, calcule su correlación con las demás variables y reporte los valores) y explique qué mide cada importancia: la caída de una métrica al barajar la variable, frente al tamaño promedio de su contribución al log-odds.

c. Con la tabla del Task 2.3, explique qué ocurre con el crédito de la variable duplicada. ¿Qué axioma de Shapley se esperaría que repartiera el crédito por igual entre la variable y su copia y por qué no lo hace en su modelo? Compare las dos configuraciones y explique por qué la dependiente de la trayectoria no es exactamente la versión condicional. ¿Qué implica esto al interpretar un valor SHAP como importancia en el mundo real y no solo en el modelo?

d. Con los resultados del Task 2.4, indique si el tiempo crece como el modelo de costo que usted propone para KernelSHAP (evaluaciones del modelo proporcionales al número de muestras por el tamaño del fondo) y si el error baja al aumentar el fondo. Luego mida cuánto tarda una predicción sobre 100 instancias y estime el tiempo para calcular los Shapley exactos de una instancia del modelo de 20 variables, que requiere $2^{20}$ evaluaciones de $v(S)$. Compare con el tiempo de KernelSHAP por instancia. Finalmente, calcule en años el tiempo que tomaría el cálculo exacto con 60 variables si cada evaluación costara un microsegundo.

---

## Task 3 (Entrega Final)

### Task 3.1

Implemente LIME para datos tabulares desde cero, sobre el modelo del Task 2.1. La representación interpretable usa las 20 variables originales: cada una está presente (conserva el valor de la instancia) o ausente (se reemplaza por un valor tomado al azar de la columna correspondiente de los datos de entrenamiento). Use los valores de esta tabla:

| Parámetro | Valor |
|---|---|
| Número de muestras $N$ | 5 000 |
| Variables que se conservan $K$ | 5 |
| Regularización $\lambda$ | 1.0 |
| Ancho relativo del kernel $\sigma_{rel}$ | 0.75 |

Siga estos pasos:

- Genere $N$ vectores binarios $z'$, cada componente igual a 1 con probabilidad 0.5, y construya la entrada $z$ correspondiente.
- La salida a explicar es la probabilidad de `good`. El peso de cada muestra es $\pi_x(z) = \exp(-D^2/\sigma^2)$, con $D^2$ el número de variables ausentes y $\sigma = \sigma_{rel} \cdot \sqrt{d'}$, con $d' = 20$.
- Ajuste primero la regresión ponderada con regularización sobre las 20 variables, usando la solución cerrada $w = (Z'^\top \Pi Z' + \lambda I)^{-1} Z'^\top \Pi y$ con una columna de unos para el intercepto (indique si lo penalizó). Conserve las $K$ variables de mayor $|w|$ y vuelva a ajustar solo con ellas.
- Calcule la fidelidad ponderada del ajuste, con $\bar{y}_w = \sum_n \pi_n y_n / \sum_n \pi_n$:

$$
R_w^2 = 1 - \frac{\sum_n \pi_n (y_n - g_n)^2}{\sum_n \pi_n (y_n - \bar{y}_w)^2}
$$

Aplique el método a tres clientes de prueba: uno con probabilidad de `good` mayor que 0.8, el rechazado del Task 2.2 y uno con probabilidad entre 0.45 y 0.55. Reporte para cada uno los coeficientes de las $K$ variables y su $R_w^2$.

### Task 3.2

Mida la estabilidad de LIME en el cliente rechazado. Para cada combinación de $N \in \{500, 2000, 5000\}$ y $\sigma_{rel} \in \{0.5, 0.75, 1.5\}$, ejecute LIME 20 veces con semillas distintas. En cada combinación calcule el índice de Jaccard promedio entre los conjuntos de las $K$ variables principales de todos los pares de corridas, $J(A,B) = |A \cap B| / |A \cup B|$, y el tamaño efectivo de muestra promedio:

$$
N_{ef} = \frac{\left(\sum_n \pi_n\right)^2}{\sum_n \pi_n^2}
$$

Presente una tabla para el Jaccard y otra para $N_{ef}$.

### Task 3.3

Compare LIME con SHAP. Realice:

a. Para los tres clientes del Task 3.1, calcule la correlación de Spearman entre $|w|$ de LIME (ajuste sobre las 20 variables) y $|\phi|$ de SHAP del Task 2.2, y la coincidencia entre las tres variables principales de cada método.

b. Para el cliente rechazado, ajuste LIME con $K \in \{3, 5, 10, 20\}$ y reporte $R_w^2$ para cada valor.

### Task 3.4

Construya explicaciones contrafactuales para el cliente rechazado del Task 2.2, con una búsqueda por muestreo y los valores de esta tabla:

| Parámetro | Valor |
|---|---|
| Número de candidatos | 20 000 |
| Máximo de variables cambiadas por candidato | 3 |
| Umbral de aprobación $\theta$ (probabilidad de `good`) | 0.5 |

Las variables `age`, `personal_status` y `foreign_worker` son inmutables. Las variables accionables son `duration`, `credit_amount`, `installment_commitment`, `checking_status` y `savings_status`. Genere los candidatos de modo que cada uno cambie un subconjunto aleatorio de entre 1 y 3 variables accionables, con valores tomados al azar de los valores observados en los datos de entrenamiento. Un candidato es válido si la probabilidad de `good` es al menos $\theta$. La distancia al original es:

$$
d(x, x') = \sum_{j \in \text{num}} \frac{|x_j - x_j'|}{\text{MAD}_j} + \sum_{j \in \text{cat}} \mathbb{1}\left[x_j \neq x_j'\right]
$$

donde $\text{MAD}_j$ es la desviación absoluta mediana de la variable $j$ en el conjunto de entrenamiento (si es cero, use la desviación absoluta media). Seleccione el candidato válido de menor distancia y luego dos más, eligiendo en cada paso el de menor distancia cuyo conjunto de variables cambiadas sea distinto de los ya elegidos. Presente una tabla con las variables cambiadas, sus valores original y nuevo (decodifique las categóricas), la distancia y la probabilidad de `good` de cada contrafactual. Reporte la validez (fracción de candidatos válidos), la proximidad promedio y la dispersión (número promedio de variables cambiadas). Si no encuentra ningún candidato válido, aumente a 5 el máximo de variables cambiadas y repórtelo.

### Task 3.5

Considere y conteste con base en sus propios resultados:

a. Con sus tablas del Task 3.2, describa cómo se relacionan $N_{ef}$ y el Jaccard. Estime cuántas muestras se necesitarían para un Jaccard de 0.9 con $\sigma_{rel} = 0.75$, justifique la extrapolación y verifíquela con una corrida adicional.

b. Identifique el cliente y la variable donde LIME y SHAP discrepan más. Proponga una hipótesis y póngala a prueba con datos: por ejemplo, reduzca $\sigma_{rel}$ y vea si la discrepancia baja, o grafique la probabilidad de `good` al variar solo esa variable en el cliente. Explique si la discrepancia se debe a la localidad, a una relación no lineal o a una interacción.

c. Con su tabla de $R_w^2$ según $K$, elija un $K$ y justifique la elección con los criterios de fidelidad y comprensibilidad vistos en clase. ¿Qué fidelidad mínima aceptaría para una decisión de crédito y por qué? ¿Qué haría si no se alcanza?

d. Compare las variables que cambian sus contrafactuales con las tres variables de mayor $|\phi|$ del mismo cliente. Identifique una variable de $|\phi|$ alto que no aparece en los contrafactuales y explique con números por qué (es inmutable, es costosa en la distancia o su efecto por unidad es pequeño). ¿Qué pregunta responde SHAP y qué pregunta responde el contrafactual?

---

## Task 4 (Entrega Final)

### Task 4.1

Valide el mapa de Grad-CAM del Task 1.2 en las tres imágenes, para la clase de mayor probabilidad, con 20 pasos y 10 órdenes aleatorios. Realice:

a. **Deletion:** ordene los píxeles de la imagen de $224 \times 224$ por el valor del mapa, de mayor a menor. En $P = 20$ pasos, reemplace progresivamente por gris medio la fracción $k/P$ de los píxeles más importantes y registre la probabilidad $c_k$ de la clase. Calcule $\text{AUC}_{del} = \frac{1}{P}\sum_{k=1}^{P} c_k$.

b. **Insertion:** parta de una versión muy desenfocada de la imagen (por ejemplo, desenfoque gaussiano con $\sigma = 10$ píxeles) e inserte progresivamente los píxeles más importantes de la imagen original. Calcule $\text{AUC}_{ins}$ de la misma forma.

c. Repita a y b con 10 órdenes aleatorios de los píxeles y reporte el promedio y la desviación estándar. Presente una tabla por imagen con las métricas del mapa y la diferencia con el azar.

### Task 4.2

Realice la prueba de aleatorización de parámetros en cascada. Reinicialice con la inicialización aleatoria de PyTorch los pesos de `fc`; luego, además, los de `layer4`; luego `layer3`, `layer2`, `layer1` y `conv1`. En cada etapa calcule el mapa de Grad-CAM de $7 \times 7$ para la clase que era la de mayor probabilidad en el modelo original (no la nueva predicción) y su correlación de Spearman con el mapa original. Presente una tabla con las tres imágenes y las seis etapas.

### Task 4.3

Considere y conteste con base en sus propios resultados:

a. Indique si Grad-CAM supera al azar en deletion e insertion y por cuántas desviaciones estándar de la distribución aleatoria. Explique la forma de sus curvas considerando que el mapa original tiene $7 \times 7$ posiciones. Luego verifique el efecto de los parches grises: repita deletion en la imagen (a) con una referencia desenfocada en lugar de gris y compare los $\text{AUC}_{del}$.

b. Indique en qué etapa de la cascada el Spearman cae por debajo de 0.3 en cada imagen. Dado que en este ResNet la derivación vista en clase da $\alpha_k^c = w_k^c / Z$, prediga qué le debería pasar al mapa cuando solo se aleatoriza `fc` y compárelo con su resultado de la primera etapa.

c. Su equipo debe entregar explicaciones del modelo de crédito del Task 2 a tres audiencias: la persona solicitante, un oficial de cumplimiento interno y un regulador. Indique, para cada una, qué explicación entregaría de las técnicas de este laboratorio, con qué validaciones respaldadas por sus números (por ejemplo, el error de eficiencia de SHAP o la estabilidad de LIME) y qué no entregaría. Explique por qué entregar un contrafactual como el del Task 3.4 sería un riesgo en un sistema de detección de lavado como el del Proyecto 2 y qué entregaría en su lugar.

---

## Entregas en Canvas

1. Documento PDF con las respuestas a cada task.
2. En la entrega parcial se espera que entreguen lo señalado. En la entrega final deben entregar **TODOS LOS TASK**.
3. Archivo `.ipynb`, o link a repositorio de GitHub (**No se acepta entregas en otros medios**).
   - a. El código debe estar comentado explicando la relación con las fórmulas de las diapositivas.

## Evaluación

1. [1.00 pt] Task 1
2. [1.00 pt] Task 2
3. [1.50 pt] Task 3
4. [0.50 pt] Task 4

**Total: 4.0 pts**
