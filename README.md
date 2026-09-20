# CC3092 — Proyecto 2: Detección de lavado de dinero en remesas

Sistema de detección de lavado de dinero en el corredor de remesas Guatemala-Estados Unidos, construido con un modelo de dos etapas:

- **Etapa A** — un autoencoder secuencial con atención que aprende qué es un patrón normal de remesas, entrenado solo sobre comportamiento legítimo.
- **Etapa B** — un clasificador que reutiliza el encoder de la Etapa A (transfer learning) para distinguir lavado de dinero de comportamiento legítimo.

Datos: [IBM AML](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) (fuente principal de secuencias por remitente) y [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (validación secundaria).

## Estructura

```
notebooks/
  proyecto2.ipynb        # notebook principal, autocontenido, ejecutado con datos reales
  checkpoints/            # pesos entrenados (Etapa A y Etapa B)
  artifacts/              # resultados exportados (ablación, métricas, datos para el MVP)
mvp/
  app.py                  # interfaz Streamlit (lee los artefactos, no ejecuta el modelo)
docs/
  Reporte_Ejecutivo_Proyecto2.tex   # reporte ejecutivo (fuente LaTeX; se compila a PDF)
  CC3092_Proyecto2.md         # enunciado del proyecto
  Documentacion_Notebook.md   # explicación sección por sección del notebook
  Uso_ia.md                   # declaración de uso de IA generativa
```

## Cómo correr el notebook

1. Sube `notebooks/proyecto2.ipynb` a [Google Colab](https://colab.research.google.com).
2. Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4).
3. Ejecutar todas las celdas. Los datos se descargan solos desde Kaggle vía `kagglehub` (no requiere clonar este repo ni instalar nada aparte).
4. Antes de cerrar la sesión de Colab, descarga las carpetas `checkpoints/` y `artifacts/` que el notebook genera — el almacenamiento de Colab es efímero.

Más detalle de qué hace cada sección en `docs/Documentacion_Notebook.md`.

## Estado del proyecto

- [x] Notebook completo, ejecutado de principio a fin con datos reales en una GPU T4 (17.8 min, dentro del límite de 30)
- [x] Ablación de 5 configuraciones × 3 semillas (ver resultados abajo)
- [x] Interpretabilidad: 5 casos analizados con pesos de atención (3 verdaderos positivos, 1 falso positivo, 1 falso negativo)
- [x] Reporte ejecutivo (`docs/Reporte_Ejecutivo_Proyecto2.tex`, se compila a PDF)
- [x] MVP funcional, desplegado en Streamlit Cloud: <https://labsdeep-proyecto2.streamlit.app/>

## Resultados

Resultado **matizado**: la señal de reconstrucción de la Etapa A mejora el ordenamiento global del riesgo, pero el experimento no demuestra que el sistema de dos etapas supere a un clasificador supervisado desde cero en la capacidad operativa de revisión.

| Configuración | AUPRC | ROC-AUC | Precision@1% |
|---|---|---|---|
| Supervisado desde cero | 0.398 | 0.830 | 0.785 |
| Solo Etapa A | 0.060 | 0.723 | 0.073 |
| Fine-tuning + reconstrucción (sistema completo) | 0.403 | 0.858 | 0.662 |

El sistema completo mejora ROC-AUC en las 3 semillas (0.858 vs 0.830), pero en AUPRC la diferencia (0.005) no es consistente, y en precision@1% el baseline es mejor. Detalle y limitaciones en el reporte y en la sección 8 del notebook.
