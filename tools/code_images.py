"""Renderiza celdas de código del notebook como imágenes (estilo captura de VS Code oscuro), como en el Lab8."""
import json, os
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter

CELLS = {"t1_1": [4, 5], "t1_2": [7], "t1_3": [9], "t1_4a": [12], "t1_4b": [14], "t1_4c": [16], "t2_1": [19],
         "t2_2a": [21], "t2_2d": [22], "t2_3": [24], "t2_4": [26], "t2_5": [28], "t2_6": [30]}

def render(nb_path="lab10.ipynb", out="figures/code"):
    os.makedirs(out, exist_ok=True)
    nb = json.load(open(nb_path, encoding="utf8"))
    for name, idxs in CELLS.items():
        for n, i in enumerate(idxs):
            src = "".join(nb["cells"][i]["source"])
            fmt = ImageFormatter(style="monokai", font_name="Consolas", font_size=17, line_numbers=False,
                                 image_pad=14, line_pad=3)
            open(f"{out}/{name}_{n}.png", "wb").write(highlight(src, PythonLexer(), fmt))
    print("imágenes de código:", sum(len(v) for v in CELLS.values()))

if __name__ == "__main__":
    render()
