# CC3092 – Deep Learning
## Laboratorio 8

## Instrucciones

Este laboratorio tiene dos partes. En la primera usted extiende el Mini-GPT de la Semana 6 con mecanismos de muestreo controlado y analiza cómo los parámetros de generación afectan la distribución del texto producido. En la segunda parte usa BERT multilingüe preentrenado para demostrar empíricamente que los embeddings contextuales capturan un tipo de información semántica que los embeddings estáticos como Word2Vec no pueden representar.

> **NOTA:** Para el Task 1 y Task 2 puede reutilizar directamente su implementación del Mini-GPT de la Semana 6. Para el Task 3 puede usar la librería `transformers` de HuggingFace para cargar los modelos preentrenados.

---

## Task 1 (Entrega Parcial)

### Task 1.1

Descargue el dataset `tiny_shakespeare` usando la siguiente URL y construya el pipeline de entrenamiento a nivel de carácter:

```
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Construya el vocabulario de caracteres únicos del texto y los mappings bidireccionales entre carácter e índice entero. El vocabulario debe tener exactamente 65 caracteres. Codifique el texto completo como una secuencia de índices y divídalo en 90% entrenamiento y 10% validación respetando el orden temporal.

Implemente una función que genere batches de secuencias de longitud fija, donde la secuencia objetivo es la secuencia de entrada desplazada un paso hacia el futuro. Use los siguientes hiperparámetros mínimos:

```
block_size = 128
d_model    = 128
n_heads    = 4
n_layers   = 3
batch_size = 64
lr         = 3e-4
epochs     = 10
```

### Task 1.2

Entrene su Mini-GPT sobre `tiny_shakespeare` con los hiperparámetros anteriores. Grafique la curva de pérdida de entrenamiento y validación por época. Al finalizar el entrenamiento calcule la **perplejidad de validación**, definida como:

$$
\text{Perplejidad} = \exp\left(\frac{1}{N}\sum_{t=1}^{N}\mathcal{L}_t\right)
$$

Donde $\mathcal{L}_t$ es la pérdida cross-entropy en el token $t$ sobre el conjunto de validación y $N$ es el número total de tokens evaluados. Un modelo bien entrenado sobre este dataset a nivel de carácter debe alcanzar una perplejidad de validación entre 3.5 y 8.0.

### Task 1.3

Implemente desde cero las tres funciones de muestreo. Las tres reciben los logits crudos del modelo y retornan el índice del siguiente token.

**Muestreo greedy:** seleccionar siempre el token con mayor logit, sin aleatoriedad.

**Muestreo top-k con temperatura:** dado un parámetro $k$ y una temperatura $\tau$, dividir los logits por $\tau$, conservar únicamente los $k$ tokens con mayor valor (asignar $-\infty$ a los demás), aplicar softmax y muestrear según esa distribución.

**Muestreo top-p con temperatura:** dado un umbral $p$ y una temperatura $\tau$, dividir los logits por $\tau$, aplicar softmax, ordenar las probabilidades de mayor a menor, incluir los tokens hasta que la probabilidad acumulada supere $p$, renormalizar y muestrear.

---

## Task 2 (Entrega Parcial)

### Task 2.1

Implemente una función `generate(prompt, max_new_tokens, strategy, **kwargs)` que genere texto carácter a carácter usando cualquiera de las tres estrategias del Task 1. El prompt es una cadena de texto que el modelo recibe como contexto inicial.

Genere 5 muestras de 200 caracteres para cada una de las siguientes configuraciones, partiendo siempre del prompt `"ROMEO:"`:

| Configuración | Estrategia | Parámetros |
|---|---|---|
| A | greedy | ninguno |
| B | top-k | k=5, tau=0.7 |
| C | top-k | k=50, tau=1.0 |
| D | top-p | p=0.70, tau=0.7 |
| E | top-p | p=0.95, tau=1.0 |

### Task 2.2

Responda las siguientes preguntas con base en el texto generado por su modelo. Sus respuestas deben hacer referencia explícita a las muestras que generó.

a. La configuración A (greedy) frecuentemente produce texto repetitivo o entra en ciclos. Explique por qué ocurre ese comportamiento usando la definición matemática del muestreo greedy y la forma de la distribución de probabilidad que el modelo aprende para el siguiente carácter.

b. Compare las configuraciones B y C (ambas top-k pero con distintos $k$ y $\tau$). Usando la fórmula de temperatura:

$$
P(w_i \mid \text{ctx}) = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}
$$

explique qué efecto tiene aumentar $k$ de 5 a 50 sobre la diversidad del texto, y qué efecto tiene cambiar $\tau$ de 0.7 a 1.0 sobre la concentración de la distribución. ¿Cuál configuración produce texto más coherente y cuál más creativo según sus muestras?

c. La configuración E (top-p=0.95, tau=1.0) es adaptativa: el número de tokens candidatos cambia en cada paso según la incertidumbre del modelo. Identifique al menos dos momentos en sus muestras donde el modelo parece estar muy seguro del siguiente carácter y dos donde parece estar inseguro. ¿Qué tipo de caracteres o posiciones en el texto corresponden a cada caso?

---

## Task 3 (Entrega Final)

### Task 3.1

Cargue `bert-base-multilingual-cased` de HuggingFace y extraiga los embeddings contextuales de la última capa para la palabra objetivo en cada una de las siguientes oraciones. Use el embedding del token correspondiente a la palabra objetivo, no el token `[CLS]`.

Las oraciones y palabras objetivo son:

| Oración | Palabra objetivo |
|---|---|
| El banco central anunció una subida de tasas de interés. | banco |
| Me senté en el banco del parque a leer el periódico. | banco |
| El banco de peces se movió velozmente ante el tiburón. | banco |
| El equipo levantó la copa del torneo ante miles de aficionados. | copa |
| Sirvió una copa de vino tinto para acompañar la cena. | copa |
| El velero desplegó su vela para aprovechar el viento. | vela |
| Encendió una vela para iluminar el cuarto durante el apagón. | vela |

Cada embedding extraído es un vector de dimensión 768.

### Task 3.2

Realice:

a. Calcule la similitud coseno entre todos los pares de embeddings de la misma palabra en contextos distintos:

$$
\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}{\lVert \mathbf{u} \rVert \cdot \lVert \mathbf{v} \rVert}
$$

Presente los resultados en una tabla con los pares de contextos para cada palabra objetivo.

b. Usando cualquier modelo Word2Vec preentrenado disponible en `gensim`, obtenga el vector estático de cada palabra objetivo y calcule la similitud coseno entre los vectores de la misma palabra en sus distintos contextos. Compare los valores obtenidos con los de BERT.

c. Visualice los embeddings de BERT con PCA reducido a 2 dimensiones. Cada punto representa una instancia de la palabra objetivo en su contexto específico. Use colores distintos para cada significado y discuta si los contextos distintos de la misma palabra se separan en el espacio de representación.

---

## Task 4 (Entrega Final)

### Task 4.1

Considere y conteste:

a. La máscara causal de GPT le impide atender a tokens futuros durante el entrenamiento. Explique por qué esa restricción es necesaria para que el objetivo de Language Modeling sea válido, y qué ocurriría si GPT usara atención bidireccional durante el preentrenamiento.

b. BERT no puede generar texto de manera autoregresiva directamente porque su atención es bidireccional. Sin embargo, los embeddings contextuales del Task 3 capturan la ambigüedad semántica de las palabras de una manera que GPT no puede hacer con la misma facilidad. Con base en sus resultados del Task 3, argumente por qué el paradigma de preentrenar y adaptar resuelve esta tensión: usted puede elegir entre GPT y BERT según la tarea que quiera resolver, sin necesidad de entrenar ninguno desde cero.

---

## Entregas en Canvas

1. Documento PDF con las respuestas a cada task.
2. En la entrega parcial se espera que entreguen lo señalado. En la entrega final deben entregar **TODOS LOS TASK**.
3. Archivo `.ipynb`, o link a repositorio de GitHub (**No se acepta entregas en otros medios**).
   - a. El código debe estar comentado explicando la relación con las fórmulas de las diapositivas.

## Evaluación

1. [1.50 pt] Task 1
2. [1.00 pt] Task 2
3. [1.00 pt] Task 3
4. [0.50 pt] Task 4

**Total: 4.0 pts**
