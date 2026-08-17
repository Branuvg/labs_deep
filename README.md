# Proyecto 1 - Semana 6: Transformer desde cero

Guía de operación, verificación y entrega del proyecto de Deep Learning. El
notebook implementa con PyTorch un encoder para clasificación de sentimiento y
un Mini-GPT para modelado de lenguaje sobre SST-2; el artefacto web usa los
pesos exportados por esos modelos para ejecutar inferencia local en el
navegador.

## Ruta rápida

1. Abra `S6 - Proyecto1_Semana6.ipynb` y confirme que todas las celdas estén
   ejecutadas en orden, sin errores, incluida la verificación final.
2. Desde la raíz del repositorio, inicie el servidor:

   ```bash
   python -m http.server 8000
   ```

3. Abra exactamente <http://localhost:8000/artifact/> y espere el estado
   `Modelos cargados · inferencia local activa`.
4. Pruebe ambos paneles y ejecute las verificaciones descritas abajo.
5. Complete los entregables humanos pendientes antes de enviar el notebook.

Detenga el servidor en la terminal con `Ctrl+C`.

## Objetivo y calificación

El proyecto vale 8 % de la nota del curso y se divide en cuatro partes:

| Parte | Resultado esperado | Puntos |
|---|---|---:|
| 1 | Transformer Encoder desde cero para clasificar sentimiento en SST-2 | 40 |
| 2 | Mini-GPT decoder-only con máscara causal para modelado de lenguaje | 25 |
| 3 | Artefacto interactivo con inferencia y mapas de atención | 15 |
| 4 | Video grupal de 3 minutos con demostración y análisis | 20 |

Las partes 1 y 2 suman los 65 puntos evaluados automáticamente. Las partes 3 y
4 son calificadas por el profesor.

## Requisitos previos

- Jupyter Notebook o un entorno compatible para abrir y ejecutar el notebook.
- Python con PyTorch, NumPy y Matplotlib para el notebook.
- Acceso a Internet al ejecutar por primera vez el Bloque 0, que descarga SST-2
  desde GitHub, y cuando `uv` necesite resolver dependencias de verificación.
- Un navegador moderno con soporte para módulos JavaScript.
- Node.js para `artifact/verify.mjs`.
- `uv` para ejecutar la comparación con PyTorch sin depender del entorno activo.

El repositorio no incluye un manifiesto de paquetes ni prescribe un comando de
instalación. Verifique las herramientas en el entorno donde trabajará en lugar
de asumir dependencias instaladas.

## Ejecución del notebook

Trabaje en `S6 - Proyecto1_Semana6.ipynb` y ejecute las celdas de arriba hacia
abajo en una misma sesión. Esto es necesario porque el corpus, vocabularios,
modelos entrenados y archivos exportados se construyen progresivamente.

- Entregue el notebook ejecutado, con salidas visibles y sin errores.
- No modifique el Bloque 0 ni el código marcado como dado, incluido el positional
  encoding.
- Implemente los bloques solicitados con tensores PyTorch puros.
- Están prohibidos `nn.MultiheadAttention`, `nn.TransformerEncoder` y cualquier
  capa de alto nivel que implemente la arquitectura completa.
- Sí se permiten `nn.Parameter`, `torch.optim.Adam`, `loss.backward()` y
  `optimizer.step()`.
- Ejecute la celda de exportación después de entrenar los modelos y la celda de
  verificación automática al final.
- La meta del grader es `65/65 pts`: Parte 1 en `40/40` y Parte 2 en `25/25`.

El estado almacenado actualmente en el notebook contiene 15 celdas de código
ejecutadas, sin salidas de error, y un subtotal automático de `65/65 pts`. Si se
reentrena o modifica el notebook, vuelva a ejecutar y revisar todas las salidas;
no dependa únicamente del resultado guardado.

## Artefacto interactivo

### Inicio correcto

Ejecute desde la raíz del repositorio, no desde `artifact/`:

```bash
python -m http.server 8000
```

Abra <http://localhost:8000/artifact/>. No abra `artifact/index.html`
directamente con una URL `file://`: los módulos del navegador usan `fetch` para
cargar `encoder_weights.json` y `gpt_weights.json`, y requieren un origen HTTP.
Servir únicamente `artifact/` tampoco funciona, porque los JSON están en el
directorio superior.

### Panel 1: análisis de sentimiento

1. Escriba una oración en inglés.
2. Seleccione **Analizar**.
3. Revise la predicción `POSITIVO` o `NEGATIVO` y la confianza softmax mostrada.
4. Compare los dos heatmaps, uno por cabeza de atención. Las filas son consultas,
   las columnas son claves y una celda más intensa representa mayor peso de
   atención. Los ejes incluyen `<CLS>`; palabras fuera del vocabulario aparecen
   como `<UNK>`.

La confianza es una salida del modelo entrenado, no una garantía de calibración
ni de calidad general.

### Panel 2: generación de texto

1. Escriba un texto semilla en inglés.
2. Ajuste la temperatura entre `0.5` y `1.5`.
3. Seleccione **Generar** y revise el texto y los dos heatmaps causales.
4. Repita con una temperatura distinta para comparar el comportamiento.

Una temperatura baja concentra la distribución y suele producir muestreo más
predecible; una temperatura alta distribuye más probabilidad y aumenta la
variación. La generación es estocástica: ejecuciones sucesivas pueden producir
textos diferentes. En los heatmaps causales, cada posición solo puede atender a
su posición y a posiciones anteriores; la atención futura debe ser cero.

## Pesos exportados

`encoder_weights.json` y `gpt_weights.json` son generados por la celda de
exportación del notebook y contienen vocabularios, dimensiones y 15 tensores por
modelo. El artefacto no entrena ni usa un servicio remoto: carga esos archivos y
reproduce en JavaScript las operaciones de los modelos.

Vuelva a generar ambos JSON después de reentrenar, cambiar pesos, vocabularios,
dimensiones o la implementación del modelo. Hágalo antes de verificar y grabar
la demostración para que el notebook y el artefacto representen la misma versión.

## Verificación local

Desde la raíz del repositorio:

```bash
node artifact/verify.mjs
uv run --with torch --with numpy python artifact/verify_parity.py
```

El primer comando valida la carga de 15 tensores por modelo, dimensiones,
predicciones de referencia, formas de atención, máscara causal y muestreo
determinista del harness. Debe terminar con líneas `OK` y sin excepciones.

El segundo ejecuta el harness JavaScript y compara logits y atención con una
referencia PyTorch. Debe informar `OK` para encoder y GPT, diferencias máximas
dentro de las tolerancias del script y atención futura máxima igual a cero. Los
valores numéricos pequeños pueden variar entre entornos; el texto interactivo no
tiene que coincidir con el texto determinista del verificador.

También compruebe manualmente que el servidor responde y que el indicador del
navegador cambia al estado de modelos cargados. La verificación actual confirma
paridad numérica entre JavaScript y PyTorch y una máscara causal estricta.

## Lista de entrega

- [ ] `S6 - Proyecto1_Semana6.ipynb` está ejecutado completamente, en orden y sin
  errores visibles.
- [ ] La verificación final del notebook conserva el objetivo de `65/65 pts`.
- [ ] La descripción del artefacto incluye una URL pública real **o** capturas de
  pantalla reales, además de la interfaz y ejemplos solicitados.
- [ ] Se documentan una oración positiva, una negativa y sus mapas de atención.
- [ ] Se documenta una generación con dos temperaturas distintas.
- [ ] Las tres preguntas de análisis tienen respuesta escrita antes del video.
- [ ] El grupo grabó un video de 3 minutos; todos los integrantes hablan y se
  muestra el artefacto, el Mini-GPT con dos temperaturas y la explicación de la
  pregunta seleccionada.
- [ ] El video se subió realmente a YouTube como no listado o a Google Drive con
  permisos correctos.
- [ ] El marcador `*[Su enlace aqui]*` de la celda **Enlace al video** fue
  reemplazado por el enlace real.
- [ ] Una búsqueda final en el notebook confirma que no quedan marcadores de
  entrega sin reemplazar.

La URL pública o las capturas, sus ejemplos finales y el video son entregables
humanos. Este repositorio no los fabrica ni demuestra que se hayan completado.
En el estado actual no hay URL pública ni capturas incluidas y el marcador del
video continúa pendiente.

## Solución de problemas

| Problema | Acción |
|---|---|
| El puerto 8000 está ocupado | Detenga el proceso que lo usa o inicie `python -m http.server 8001` y abra `http://localhost:8001/artifact/`. |
| Los pesos devuelven 404 | Confirme que ambos JSON existen en la raíz y que el servidor se inició desde esa raíz, no desde `artifact/`. |
| La página muestra un error de carga | Revise la terminal del servidor y la consola del navegador; no use `file://`. |
| `python` no encuentra archivos o la URL lista otro contenido | Vuelva a la carpeta que contiene el notebook, los JSON y `artifact/` antes de iniciar el servidor. |
| Falta `torch` al ejecutar la paridad | Use exactamente `uv run --with torch --with numpy python artifact/verify_parity.py`; requiere `uv` y puede requerir Internet para resolver paquetes. |
| El navegador conserva pesos o código anteriores | Haga una recarga forzada o limpie la caché del sitio después de regenerar los JSON. |

## Estado verificado y pendientes

- Grader almacenado en el notebook: `65/65 pts` (`40/40` encoder y `25/25`
  Mini-GPT).
- Artefacto: paridad numérica JavaScript/PyTorch verificada y atención causal
  futura igual a cero.
- Pendiente humano: URL pública real o capturas reales del artefacto.
- Pendiente humano: video grupal y reemplazo del marcador por su enlace real.

Este estado describe los archivos actuales; vuelva a verificar después de
cualquier reentrenamiento o cambio.

## Estructura del proyecto

```text
.
├── S6 - Proyecto1_Semana6.ipynb  # Implementación, entrenamiento y entrega
├── encoder_weights.json          # Pesos exportados del encoder
├── gpt_weights.json              # Pesos exportados del Mini-GPT
├── sst2_train.tsv                # Corpus de entrenamiento descargado
├── sst2_dev.tsv                  # Corpus de validación descargado
├── convergencia_encoder.png      # Curva guardada del encoder
├── convergencia_gpt.png          # Curva guardada del Mini-GPT
├── artifact/
│   ├── index.html                # Interfaz de los dos paneles
│   ├── app.mjs                   # Interacción y renderizado de heatmaps
│   ├── model.mjs                 # Inferencia Transformer en JavaScript
│   ├── styles.css                # Presentación visual
│   ├── verify.mjs                # Verificación funcional en Node.js
│   ├── verify_parity.py          # Comparación JavaScript/PyTorch
│   └── README.md                 # Referencia breve del artefacto
└── README.md                     # Esta guía de operación y entrega
```
