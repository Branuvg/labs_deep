# lab10_deep

Laboratorio 10 (CC3045 Deep Learning): explicabilidad con Grad-CAM/CAM (ResNet-18) y SHAP (German Credit).

- `lab10.ipynb`: código de los Tasks 1 y 2 (semilla 42). Se genera con `tools/build_notebook.py`.
- `Lab10_DL.docx`: respuestas (se genera con `tools/fill_docx.py` a partir de `tools/Lab10_DL_plantilla.docx`).
- `images/`, `figures/`, `results/`, `data/credit-g.arff` (respaldo local del dataset de OpenML).

## Entorno
```
py -3.12 -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name lab10 --display-name "lab10 (venv)"
```
