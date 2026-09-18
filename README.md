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
docs/
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

- [x] Notebook completo, ejecutado de principio a fin con datos reales
- [x] Ablación de 5 configuraciones × 3 semillas — confirma que la arquitectura de dos etapas aporta valor sobre un clasificador supervisado desde cero
- [x] Interpretabilidad: casos analizados con pesos de atención
- [ ] Reporte ejecutivo (PDF)
- [ ] MVP funcional
