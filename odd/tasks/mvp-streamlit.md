# MVP funcional — Streamlit Cloud (Componente 4, Proyecto 2)

## Objetivo
Construir e implementar la interfaz interactiva exigida por el Componente 4 de
`docs/CC3092_Proyecto2.md` (15 pts), desplegable en Streamlit Cloud sin que el
evaluador instale nada.

## Problema
Los artefactos del modelo ya existen (`notebooks/artifacts/`), pero no hay
interfaz. Un notebook local no cuenta como MVP y un MVP que no corre vale cero.

## Alcance autorizado
- Crear `mvp/app.py`, `mvp/requirements.txt`, `mvp/README.md`.
- Leer artefactos ya versionados. No reentrenar, no modificar el notebook.

## Restricciones
- Sin inferencia en vivo: la app consume `mvp_data.json` precomputado.
- Umbral Etapa A = `official_threshold` de `metrics.json` (1.2921).
- Umbral Etapa B no fue exportado; se deriva de `pred` (min de `prob_b` con
  `pred == 1`), lo que reproduce exactamente las decisiones del notebook.
- Copy de UI en español neutro (el proyecto y su lector objetivo son en español).

## Modo TDD
Desactivado. No hay suite de tests en el repositorio ni runner configurado;
la verificación es funcional (arranque de la app y chequeo de artefactos).

## Estrategia de entrega
`single-pr` — un solo alcance acotado sobre la rama `proyecto2`.

## Tareas
- [x] T1 — `mvp/app.py`: carga cacheada de artefactos, selector de remitente,
      tabla de secuencia, scores de ambas etapas, mapa de calor de atención y
      párrafo explicativo automático.
- [x] T2 — `mvp/requirements.txt` y `mvp/README.md` con pasos de despliegue.
- [x] T3 — Verificación funcional: la app arranca sin errores y renderiza un
      remitente alertado y uno no alertado.

## Criterios de aceptación
Los cinco requisitos del Componente 4 son visibles en la interfaz y la app
corre sin excepciones desde una copia limpia del repositorio.

## Evidencia
- `streamlit.testing.v1.AppTest` sobre `mvp/app.py` con los cuatro filtros de la
  barra lateral: `Todos`, `Solo alertados`, `Falsos negativos` y
  `Solo lavado confirmado`. Resultado: cero excepciones en los cuatro.
- Caso alertado (`808839F70`): Etapa A 0.728, Etapa B 97.8%, decision ALERTA,
  etiqueta real Lavado.
- Caso sin alerta (`801812250`, falso negativo): Etapa A 0.800, Etapa B 33.5%,
  decision Sin alerta, etiqueta real Lavado.
- Umbral de Etapa B derivado de `pred`: 0.3399, consistente con la tasa de
  alertas objetivo de 1%.
- Dependencias instaladas desde `mvp/requirements.txt` en un entorno limpio.

## Notas
- Los pins exactos de `pandas` fallaron al compilar en Python 3.14; se pasaron a
  cotas inferiores para que pip resuelva wheels en cualquier interprete soportado.
- `use_container_width` esta deprecado en Streamlit 1.64; se usa `width="stretch"`,
  por eso el piso de version es `streamlit>=1.49`.

## Siguiente paso
Desplegar en Streamlit Cloud (requiere repositorio publico y la rama publicada)
y guardar la URL en el `.txt` de entrega.
