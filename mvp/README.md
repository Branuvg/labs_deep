# MVP funcional — Detección de lavado de dinero en remesas

Interfaz interactiva del Componente 4 de CC3092 Proyecto 2.

## Qué muestra

- Selección de un remitente del conjunto de prueba, con filtros por alertados,
  lavado confirmado y falsos negativos.
- La secuencia completa de transacciones con monto, tipo, hora, salto temporal,
  destino y si el destino es nuevo.
- El score de anomalía de la Etapa A y la probabilidad de lavado de la Etapa B,
  cada uno contrastado contra su umbral operativo.
- Un mapa de calor de los pesos de atención `alpha` sobre la secuencia, que
  identifica qué transacciones sustentan la alerta.
- Un párrafo de explicación en lenguaje natural redactado automáticamente.

## Cómo corre

La app **no ejecuta el modelo**: consume los artefactos que el notebook exportó
en `notebooks/artifacts/` (`mvp_data.json`, `metrics.json`,
`preprocessing.json`). Por eso no necesita PyTorch ni GPU y arranca en segundos.

## Ejecución local

```bash
pip install -r mvp/requirements.txt
streamlit run mvp/app.py
```

## Despliegue en Streamlit Cloud

1. Asegurarse de que la rama esté publicada en GitHub y que el repositorio sea
   público (Streamlit Cloud gratuito no lee repos privados).
2. Entrar a <https://share.streamlit.io> e iniciar sesión con GitHub.
3. **New app** y completar:
   - Repository: `<usuario>/<repositorio>`
   - Branch: `proyecto2`
   - Main file path: `mvp/app.py`
4. **Deploy**. El primer arranque instala dependencias y tarda un par de minutos.
5. Copiar la URL pública resultante al archivo `.txt` de entrega.
