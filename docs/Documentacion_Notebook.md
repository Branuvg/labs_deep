# Documentación del notebook — `notebooks/proyecto2.ipynb`

Explicación detallada de qué hace cada sección del notebook, cómo lo hace, y qué necesita. Sirve como referencia rápida para escribir el reporte ejecutivo (Componente 3) sin tener que releer el código completo.

## Qué es y qué no es este documento

Esto documenta **el notebook**, no el proyecto completo. El reporte PDF, el MVP, y las decisiones de negocio están fuera de alcance aquí — para eso está `docs/CC3092_Proyecto2.md` (el enunciado) y el propio reporte.

## Requisitos para correrlo

- **Entorno:** Google Colab con GPU T4 (gratuita). Corre en CPU también, pero mucho más lento (~30-50 min en CPU vs. ~19 min confirmado en T4).
- **Dependencias:** solo `kagglehub` se instala explícitamente (la celda de bootstrap lo hace sola si detecta Colab); todo lo demás (`torch`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`) ya viene en el entorno de Colab.
- **Datos:** se descargan solos desde Kaggle vía `kagglehub`, de forma anónima (no se necesitó `kaggle.json` en las pruebas). Sube ~475MB de IBM AML + ~493MB de PaySim.
- **No requiere clonar ningún repo ni tener el resto de este proyecto disponible** — el notebook es 100% autocontenido.

## Estructura general

El notebook tiene 34 celdas de código y 21 de markdown (55 en total), organizadas en 11 secciones. Cada decisión de diseño no obvia está documentada en una celda markdown con el formato **Decisión / Justificación / Alternativa descartada** — hay 8 de esas celdas repartidas en el notebook; son la fuente más rápida para entender *por qué* se hizo cada cosa, no solo *qué* se hizo.

---

## Sección 1 — Configuración y entorno

**Qué hace:** define `CFG`, un diccionario único con **todos** los hiperparámetros del proyecto (nada de números sueltos en el código). Detecta si está en Colab (instala `kagglehub` si hace falta), fija las semillas de aleatoriedad (`torch`, `numpy`, `random`) y activa determinismo total con `torch.use_deterministic_algorithms(True)`.

**Hiperparámetros clave y por qué:**

| Parámetro | Valor | Por qué |
|---|---|---|
| `SEQUENCE_SOURCE` | `"ibm_aml"` | ver Sección 2 — PaySim no sirve para secuencias por remitente |
| `MAX_LEN` | 32 (se recalcula en Sección 3) | valor inicial; el real sale del percentil 90 de longitud de secuencia |
| `MIN_LEN` | 5 | remitentes con menos de 5 transacciones no forman una secuencia útil |
| `N_SENDERS` | 60,000 | subconjunto manejable en 30 min; se prioriza quedarse con **todos** los positivos |
| `HIDDEN`, `LATENT` | 64, 64 | tamaño de la GRU y del cuello de botella del autoencoder |
| `EPOCHS_A`, `EPOCHS_B` | 15, 12 | épocas de Etapa A (autoencoder) y Etapa B (clasificador) |
| `FREEZE_EPOCHS` | 1 | cuántas épocas de Etapa B se entrenan con el encoder congelado antes de descongelar |
| `LR_HEAD`, `LR_ENCODER` | 1e-3, 3e-4 | learning rate diferencial: la cabeza nueva aprende más rápido que el encoder preentrenado |
| `FOCAL_GAMMA`, `FOCAL_ALPHA` | 2.0, 0.25 | hiperparámetros de la focal loss (ver Sección 7) |
| `ALERT_RATE` | 0.01 (1%) | capacidad operativa asumida del equipo de cumplimiento |
| `SEEDS` | `[0, 1, 2]` | tres semillas para reportar media ± desviación estándar en la ablación |

**Qué necesita:** nada externo; es la primera celda que corre.

---

## Sección 2 — Carga de datos y diagnóstico de la fuente

**Qué hace:** descarga (o reutiliza caché) el archivo de transacciones de IBM AML (`HI-Small_Trans.csv`, ~475MB) y de PaySim, normaliza ambos a un esquema común (`sender_id, timestamp, amount, tx_type, dest_id, label`), y corre un diagnóstico obligatorio: distribución de transacciones por remitente y por destinatario.

**Cómo lo hace:** `get_raw_data_path()` usa `kagglehub.dataset_download(handle, path="archivo.csv")` para bajar **solo el archivo necesario** de IBM AML (no las ~7.6GB de las 6 variantes del dataset completo). Para PaySim se baja el dataset completo porque solo trae un CSV (sin ambigüedad) y la descarga selectiva por archivo demostró devolver un blob comprimido sin descomprimir para ese dataset específico (documentado en la celda de Decisión correspondiente).

**Por qué IBM AML y no PaySim como fuente principal** (aunque el enunciado lo llama "dataset principal"): el diagnóstico confirma que en PaySim `nameOrig` es casi único por fila (media ≈1.001 transacciones por remitente) — es literalmente imposible construir una secuencia temporal de comportamiento con eso. IBM AML sí tiene remitentes recurrentes (hasta 168,672 transacciones para una cuenta). PaySim se mantiene cargado como validación secundaria, no se descarta.

**Qué necesita:** conexión a internet (Kaggle). Nada más.

---

## Sección 3 — Ingeniería de features y construcción de secuencias

**Qué hace:** calcula 7 features por transacción y arma secuencias por remitente.

| Feature | Qué mide |
|---|---|
| `log_amount` | monto (log1p, por las colas pesadas) |
| `delta_t` | horas desde la transacción anterior del mismo remitente (log1p) |
| `hour_sin`, `hour_cos` | hora del día, codificada cíclicamente |
| `tx_type` (one-hot) | tipo de operación |
| `is_new_dest` | 1 si el destino no había aparecido antes en la secuencia de ese remitente |
| `dest_entropy` | entropía acumulada de destinos hasta ese punto (mide "abanico" de contrapartes) |
| `threshold_ratio` | monto / $10,000 (fraccionamiento bajo umbral regulatorio) |

**Cómo construye las secuencias:** agrupa por `sender_id`, ordena por tiempo, descarta remitentes con menos de `MIN_LEN=5` transacciones (documentando cuántos positivos se pierden por ese filtro), trunca a las últimas `MAX_LEN` transacciones (donde `MAX_LEN` sale del percentil 90 real de la distribución de longitudes, no de un número fijo), y hace padding a la izquierda con una máscara booleana. Si hay más remitentes que `N_SENDERS`, hace un submuestreo que **conserva todos los positivos** y recorta negativos al azar.

**Detalle de rendimiento importante:** `dest_entropy` originalmente se calculaba de forma ingenua; el cálculo actual usa una fórmula incremental exacta ($H = \log N - S/N$ con $S=\sum n_i\log n_i$ actualizado en O(1) por paso) para que no explote con cuentas de miles de transacciones.

**Limitación conocida sobre `MAX_LEN`:** el percentil 90 se calcula sobre la longitud de **todos** los remitentes (incluyendo el ~74% que luego se descarta por `MIN_LEN=5`), no solo sobre los que sobreviven el filtro. Eso arrastra el percentil hacia abajo: en la corrida real dio `MAX_LEN=29` en vez de un valor más representativo (~64 si se calculara solo sobre remitentes válidos). Sigue siendo data-driven y no arbitrario, pero es una mejora pendiente documentada en la celda de Decisión correspondiente, no corregida para no tener que volver a correr todo el pipeline (~20-30 min) otra vez.

**Qué necesita:** el DataFrame ya cargado de la Sección 2.

---

## Sección 4 — Splits y verificación de fuga

**Qué hace:** separa train/val/test **por remitente** (nunca por transacción), 70/15/15, estratificado por etiqueta. Incluye un `assert` explícito de que ningún `sender_id` aparece en más de un split — si algún día se rompe, el notebook falla ruidosamente en vez de dar resultados contaminados. También compara la distribución temporal de cada split (documentando el riesgo de fuga temporal que un split puramente aleatorio no elimina del todo). La normalización (media/desviación) se calcula **solo sobre train** y se aplica a los tres splits.

**Qué necesita:** las secuencias `X, mask, y, sender_ids` de la Sección 3.

---

## Sección 5 — Visualizaciones exigidas (Componente 1)

**Qué hace:** genera las tres verificaciones que pide el enunciado antes de modelar: histograma de longitud de secuencia, tabla de proporción positivos/negativos por split, y 3 secuencias normales vs. 3 sospechosas como heatmap (features × tiempo).

**Qué necesita:** los splits de la Sección 4.

---

## Sección 6 — Etapa A: autoencoder secuencial con atención

**Qué hace:** entrena un autoencoder que aprende a reconstruir el comportamiento **normal** (solo remitentes negativos de train). El error de reconstrucción se convierte en el score de anomalía.

**Arquitectura (`SeqAutoencoder`):**
- `SeqEncoder`: GRU bidireccional → `AttnPool` (atención aditiva tipo Bahdanau) → proyección a un vector latente `z` de tamaño `LATENT`.
- `SeqDecoder`: **autoregresivo con teacher forcing** — en cada paso recibe `[z, x_{t-1}]` (la transacción real anterior), no solo `z` repetido. Esto es un rediseño respecto a una primera versión que sí repetía `z` de forma constante; esa versión aprendía una reconstrucción casi trivial (MSE de validación estancado cerca de 1.0, el nivel de "predecir siempre la media" con features normalizadas). Con el decoder autoregresivo, el MSE de validación baja genuinamente (de ~0.81 a ~0.45 en 15 épocas).
- Pérdida: MSE enmascarado (se divide entre posiciones válidas, no entre `L`, para no penalizar de más las secuencias cortas).

**Umbral de anomalía:** se calculan dos criterios sobre validación — el que maximiza F₂ (prioriza recall) y el que corresponde a `ALERT_RATE=1%` (capacidad operativa real) — y se reportan ambos en una tabla. El oficial es el de `ALERT_RATE`, porque el problema de negocio (según el enunciado) es precisamente que los sistemas actuales generan más alertas de las que un equipo puede revisar.

**Qué necesita:** los splits normalizados (`X_norm`) de la Sección 4.

---

## Sección 7 — Etapa B: transfer learning y pérdida

**Qué hace:** construye `Detector`, un clasificador que reutiliza el encoder de la Etapa A (no aprende desde cero) para distinguir lavado de comportamiento legítimo.

**Cómo transfiere:** `FREEZE_EPOCHS` épocas con el encoder congelado (`requires_grad=False`, solo entrena la cabeza), luego se descongela y se entrena con **learning rate discriminativo** (`LR_ENCODER` para el encoder, más bajo; `LR_HEAD` para la cabeza, más alto) — mitiga el *catastrophic forgetting*.

**Función de pérdida:** focal loss implementada a mano (no importada), justificada por el desbalance extremo de clases (~1.3% de positivos).

**Cómo combina las dos etapas:** el error de reconstrucción de la Etapa A entra como **feature adicional** a la cabeza del clasificador (`concat([z, recon_error_normalizado])`), no como un promedio de scores en escalas distintas.

**Qué necesita:** el encoder y decoder ya entrenados de la Etapa A.

---

## Sección 8 — Experimento de ablación (obligatorio)

**Qué hace:** compara 5 configuraciones, cada una con las 3 semillas de `CFG["SEEDS"]`, para demostrar empíricamente si la arquitectura de dos etapas aporta valor:

1. Supervisado desde cero (sin Etapa A) — la línea base.
2. Solo Etapa A (el error de reconstrucción como único score).
3. Encoder congelado todo el entrenamiento (feature extraction puro).
4. Fine-tuning completo (sin la feature de reconstrucción).
5. Fine-tuning completo + feature de reconstrucción — el sistema completo.

**Métricas:** AUPRC (principal, justificado por el desbalance extremo — ROC-AUC se vuelve poco informativo cuando hay tantísimos más negativos que positivos), `precision@ALERT_RATE`, `recall@ALERT_RATE`, F₂, y ROC-AUC como referencia secundaria.

**Resultado real (última corrida validada, ver `artifacts/ablation.csv`):** la configuración 5 (sistema completo) supera al baseline supervisado en las **3 semillas individualmente** en AUPRC y ROC-AUC — el experimento demuestra que la arquitectura de dos etapas sí aporta valor. Las configuraciones 2 y 3 (sin fine-tuning) rinden mucho peor, lo que confirma que el fine-tuning completo es la parte que realmente importa, no solo tener un encoder preentrenado.

**Qué necesita:** vuelve a entrenar la Etapa A por cada semilla (para no reusar el mismo modelo en las 3 semillas) y luego entrena la Etapa B para cada una de las configuraciones aplicables.

---

## Sección 9 — Interpretabilidad y casos

**Qué hace:** selecciona automáticamente 5 casos del conjunto de prueba (3 verdaderos positivos con mayor confianza, 1 falso positivo, 1 falso negativo), grafica la secuencia con los pesos de atención (`alpha`) superpuestos como mapa de calor, y aplica una función heurística (`tipologia()`) que intenta mapear el patrón atendido a una tipología conocida de lavado ("fraccionamiento bajo umbral", "velocidad inusual", "abanico de destinos", "concentración horaria").

**Verificación incluida:** un `assert` de que `alpha` suma 1 sobre las posiciones válidas de cada secuencia (la atención está bien normalizada).

**Limitación conocida y documentada:** la heurística de tipología es imprecisa — su primera condición (`threshold_ratio > 0.8`, monto > $8,000) dispara con cualquier transacción grande, no solo con montos agrupados justo debajo de un umbral regulatorio. En la corrida real, los 5 casos seleccionados recibieron la misma etiqueta por esta razón. El análisis textual de cada caso (en la celda markdown inmediatamente después) corrige esto manualmente, interpretando los números reales de atención en vez de confiar ciegamente en la heurística.

**Qué necesita:** el modelo final de la configuración 5 (el ganador de la ablación) y los datos de test.

---

## Sección 10 — Exportación de artefactos

**Qué hace:** escribe los 6 archivos que el resto del proyecto (reporte, MVP) necesita:

| Archivo | Contenido |
|---|---|
| `checkpoints/stage_a.pt` | pesos del autoencoder |
| `checkpoints/stage_b.pt` | pesos del detector final |
| `artifacts/preprocessing.json` | `CFG`, estadísticas de normalización, nombres de features |
| `artifacts/metrics.json` | tabla de ablación, umbrales elegidos |
| `artifacts/ablation.csv` | la tabla de ablación en crudo |
| `artifacts/mvp_data.json` | un registro por remitente de test: transacciones, score de anomalía, probabilidad, atención, tipología — todo lo que el MVP necesita para mostrar una alerta explicada |

**Detalle de rendimiento importante:** la construcción de `mvp_data.json` originalmente filtraba el DataFrame completo de transacciones por cada remitente de test dentro de un loop (`df_valid[df_valid["sender_id"]==sid]`), lo cual es O(n × n_test) y con `sender_id` de tipo texto llegó a tomar **~17 minutos** en una corrida real. Se reemplazó por un único `groupby("sender_id")` antes del loop, con búsquedas O(1) después — la misma operación pasó a tomar ~1 segundo. Este fue el ajuste que permitió bajar de 46.9 min a 19.3 min en Colab con GPU T4.

Si `mvp_data.json` supera 25MB, se submuestrea a 2,000 remitentes (conservando **todos** los positivos) para que quepa embebido en un artifact de Claude si esa es la ruta elegida para el MVP.

**Dónde aparecen físicamente `checkpoints/` y `artifacts/`:** son rutas relativas (`os.makedirs("checkpoints", ...)`), así que se crean en el directorio de trabajo desde donde corre el kernel — **no necesariamente donde tú crees**. En Jupyter/Colab eso normalmente es el mismo directorio donde está el `.ipynb` (en Colab, por defecto `/content/`, junto al notebook subido). Ahora mismo, de la corrida de referencia, ya existen físicamente en `notebooks/checkpoints/` y `notebooks/artifacts/` junto a `proyecto2.ipynb` — no hace falta correr nada para verlos.

**Cómo recuperarlos después de correr en Colab (importante, es fácil perderlos):** el almacenamiento de una sesión de Colab es **efímero** — vive solo mientras el runtime esté activo, y se borra por completo cuando se desconecta o se recicla (por inactividad, o al cerrar la pestaña). Si corres el notebook en Colab, antes de cerrar la sesión tienes que bajarte `checkpoints/` y `artifacts/` manualmente: panel de archivos a la izquierda (ícono de carpeta) → clic derecho sobre cada carpeta → "Download" (Colab las comprime a `.zip` automáticamente), o móntalos a tu Google Drive con `from google.colab import drive; drive.mount('/content/drive')` y copia los archivos ahí antes de terminar. Si cierras la sesión sin hacer esto, **hay que volver a correr todo el notebook para regenerarlos** — no hay atajo.

**Qué necesita:** todo lo anterior — es la última sección computacional.

---

## Sección 11 — Conexión con el curso

**Qué hace:** conecta explícitamente cada técnica usada con la semana del curso donde se estudió (S3 RNN/LSTM → por qué GRU y no RNN vanilla; S4 Encoder-Decoder/Autoencoders → la Etapa A y el rediseño del decoder; S5 Attention → `AttnPool`; S8 Pérdidas/Regularización/Optimización → focal loss, AUPRC, Adam, dropout; S9 Transfer Learning → la estrategia de congelamiento + fine-tuning + learning rate diferencial). Esta sección está completa, usando `deep-learning-compendio.md` como referencia de contenido del curso.

---

## Decisiones de diseño documentadas (formato Decisión/Justificación/Alternativa)

Hay 8 celdas de este tipo en el notebook, en este orden: (1) IBM AML vs. PaySim como fuente principal, (2) descarga selectiva de archivos vs. dataset completo, (3) truncar secuencias por percentil 90, (4) umbral oficial de Etapa A basado en `ALERT_RATE`, (5) decoder autoregresivo vs. repetir `z`, (6) hiperparámetros de la focal loss, (7) reconstrucción como feature concatenada vs. promedio de scores. Cada una sigue el mismo patrón: qué se decidió, por qué (con evidencia cuando la hay, como los números reales de la curva de pérdida), y qué alternativa se descartó y por qué.

## Qué queda pendiente (fuera de este notebook)

- El reporte ejecutivo en PDF (Componente 3) y el MVP funcional (Componente 4) son entregables separados que consumen los artefactos de la Sección 10, pero no se construyen en este notebook.
- La sección de uso de IA generativa se documenta aparte, en `docs/Uso_ia.md`.
- Los nombres del grupo van en el reporte, no en el notebook.
