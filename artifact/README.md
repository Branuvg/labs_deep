# Artifact interactivo: Transformer desde cero

Visualización local de los pesos entrenados y exportados por `S6 - Proyecto1_Semana6.ipynb`. El navegador ejecuta directamente el encoder de sentimiento y el Mini-GPT; no usa un servicio remoto ni respuestas simuladas.

## Ejecución

Desde la raíz del repositorio:

```bash
python -m http.server 8000
```

Abra `http://localhost:8000/artifact/`. Es necesario servir la raíz completa porque el artifact carga `encoder_weights.json` y `gpt_weights.json` desde el directorio superior.

## Verificación

```bash
node artifact/verify.mjs
uv run --with torch --with numpy python artifact/verify_parity.py
```

El primer comando valida carga, formas, máscara causal y muestreo determinista. El segundo compara numéricamente la implementación JavaScript con una referencia PyTorch que reproduce las operaciones del notebook.
