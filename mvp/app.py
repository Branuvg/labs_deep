"""MVP funcional del sistema de deteccion de lavado en remesas.

Componente 4 de CC3092 Proyecto 2. La interfaz no reentrena ni ejecuta el
modelo: consume los artefactos precomputados que el notebook exporto en
`notebooks/artifacts/`, de modo que el despliegue no requiere PyTorch ni GPU.

Cada remitente del conjunto de prueba trae:
  - `anomaly_score_a`: error de reconstruccion del autoencoder de Etapa A,
    es decir  ||x - x_hat||^2  promediado sobre las posiciones validas.
  - `prob_b`: salida sigmoide del clasificador de Etapa B,  P(lavado | secuencia).
  - `attention`: vector alpha de la capa de atencion, con  sum(alpha) = 1  sobre
    las transacciones validas. Es la contribucion relativa de cada transaccion
    al vector de contexto que alimenta la cabeza de clasificacion.
"""

from pathlib import Path
import json

import altair as alt
import pandas as pd
import streamlit as st

# Los artefactos viven junto al notebook; la app se ejecuta desde la raiz del
# repositorio en Streamlit Cloud, asi que la ruta se resuelve desde este archivo.
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "artifacts"

TOP_K_EXPLICACION = 3

# Lectura en lenguaje de negocio de cada tipologia que produce la heuristica
# `tipologia()` del notebook, para que el parrafo explicativo sea entendible
# por un oficial de cumplimiento.
TIPOLOGIA_GLOSA = {
    "fraccionamiento bajo umbral": (
        "varios envios con montos apenas por debajo del umbral de reporte, "
        "patron tipico de fraccionamiento (structuring)"
    ),
    "velocidad inusual": (
        "una rafaga de transacciones muy juntas en el tiempo, inconsistente "
        "con el ritmo habitual del remitente"
    ),
    "abanico de destinos": (
        "dispersion del dinero hacia muchos destinatarios nuevos, patron "
        "compatible con una red de mulas"
    ),
    "concentracion horaria": (
        "actividad concentrada en una franja horaria muy estrecha, poco "
        "habitual para un remitente legitimo"
    ),
    "sin patron claro": (
        "sin una tipologia conocida dominante; la alerta responde al perfil "
        "global de la secuencia"
    ),
}


@st.cache_data(show_spinner="Cargando artefactos del modelo...")
def cargar_artefactos():
    """Lee los artefactos exportados por el notebook una sola vez por sesion."""
    with open(ARTIFACTS_DIR / "mvp_data.json") as f:
        registros = json.load(f)
    with open(ARTIFACTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    with open(ARTIFACTS_DIR / "preprocessing.json") as f:
        preprocessing = json.load(f)
    return registros, metrics, preprocessing


def umbral_etapa_b(registros, metrics):
    """Umbral de decision de Etapa B.

    Los artefactos nuevos lo exportan como `threshold_b` (cuantil 1 - ALERT_RATE
    de las probabilidades de validacion) y se usa tal cual. Si el artefacto es
    anterior y no lo trae, se recupera a partir de `pred`: el JSON esta
    submuestreado, asi que recalcular un cuantil daria otro valor, y la menor
    probabilidad marcada como alerta reproduce las decisiones del notebook.
    """
    if "threshold_b" in metrics:
        return metrics["threshold_b"]
    alertados = [r["prob_b"] for r in registros if r["pred"] == 1]
    return min(alertados) if alertados else 1.0


def tabla_secuencia(registro):
    """Arma la secuencia de transacciones con su peso de atencion asociado."""
    txs = registro["transactions"]
    attn = registro["attention"]
    # Defensa: si por truncamiento las longitudes difieren, se alinea al minimo.
    n = min(len(txs), len(attn))
    filas = []
    for i in range(n):
        tx = txs[i]
        filas.append({
            "#": tx["idx"],
            "Monto (USD)": tx["amount"],
            "Tipo": tx["type"],
            "Hora": tx["hour"],
            "Horas desde la anterior": round(tx["delta_t_h"], 2),
            "Destino": tx["dest_id"],
            "Destino nuevo": "si" if tx["is_new_dest"] else "no",
            "Atencion": attn[i],
        })
    return pd.DataFrame(filas)


def generar_explicacion(registro, df_seq, thr_a, thr_b):
    """Redacta el parrafo de explicacion en lenguaje natural.

    Combina tres senales: el score de Etapa A frente a su umbral, la
    probabilidad de Etapa B frente al suyo, y las transacciones que
    concentran la mayor masa de atencion alpha.
    """
    alerta = registro["pred"] == 1
    score_a = registro["anomaly_score_a"]
    prob_b = registro["prob_b"]

    top = df_seq.nlargest(TOP_K_EXPLICACION, "Atencion")
    masa = top["Atencion"].sum()
    ids = ", ".join(f"#{int(i)}" for i in top["#"])
    glosa = TIPOLOGIA_GLOSA.get(registro["tipologia"], registro["tipologia"])

    if alerta:
        # La decision la toma la Etapa B. Cuando la Etapa A queda por debajo de
        # su umbral se dice explicitamente, porque para un oficial de
        # cumplimiento es relevante saber que la deteccion vino del componente
        # supervisado y no de la senal de anomalia no supervisada.
        if score_a >= thr_a:
            contexto_a = (
                f"La Etapa A ya lo marcaba como anomalo, con un score de "
                f"{score_a:.3f} sobre un umbral operativo de {thr_a:.3f}"
            )
        else:
            contexto_a = (
                f"La Etapa A por si sola no lo habria marcado (score de "
                f"anomalia {score_a:.3f} frente a un umbral de {thr_a:.3f}), "
                f"pero la senal decisiva viene del clasificador supervisado"
            )
        apertura = (
            f"El remitente **{registro['sender_id']}** genera **alerta**. "
            f"{contexto_a}: la Etapa B estima una probabilidad de lavado de "
            f"{prob_b:.1%}, superior al umbral de decision de {thr_b:.1%}."
        )
    else:
        apertura = (
            f"El remitente **{registro['sender_id']}** **no genera alerta**. "
            f"La Etapa A le asigna un score de anomalia de {score_a:.3f} "
            f"frente a un umbral de {thr_a:.3f}, y la Etapa B estima una "
            f"probabilidad de lavado de {prob_b:.1%}, por debajo del umbral "
            f"de decision de {thr_b:.1%}."
        )

    cuerpo = (
        f" El modelo concentra el {masa:.0%} de su atencion en las "
        f"transacciones {ids} de un total de {len(df_seq)}. "
        f"La mayor de ellas es un envio de USD {top.iloc[0]['Monto (USD)']:,.2f} "
        f"por {top.iloc[0]['Tipo']} a las {int(top.iloc[0]['Hora'])}:00 hacia un "
        f"destino {'nuevo' if top.iloc[0]['Destino nuevo'] == 'si' else 'ya conocido'}. "
        f"El patron atendido corresponde a {glosa}."
    )

    cierre = (
        " Recomendacion: escalar el caso a revision manual con prioridad."
        if alerta else
        " Recomendacion: mantener en monitoreo rutinario, sin accion inmediata."
    )
    return apertura + cuerpo + cierre


def main():
    st.set_page_config(page_title="Deteccion de lavado en remesas", layout="wide")
    st.title("Deteccion de lavado de dinero en remesas")
    st.caption(
        "Sistema de dos etapas: autoencoder secuencial con atencion (Etapa A) "
        "y clasificador supervisado por transfer learning (Etapa B)."
    )

    registros, metrics, preprocessing = cargar_artefactos()
    thr_a = metrics["official_threshold"]
    thr_b = umbral_etapa_b(registros, metrics)

    # ---- Seleccion del remitente -------------------------------------------
    st.sidebar.header("Seleccion de remitente")
    filtro = st.sidebar.radio(
        "Filtrar el conjunto de prueba",
        ["Todos", "Solo alertados", "Solo lavado confirmado", "Falsos negativos"],
    )
    if filtro == "Solo alertados":
        pool = [r for r in registros if r["pred"] == 1]
    elif filtro == "Solo lavado confirmado":
        pool = [r for r in registros if r["label"] == 1]
    elif filtro == "Falsos negativos":
        pool = [r for r in registros if r["label"] == 1 and r["pred"] == 0]
    else:
        pool = registros

    if not pool:
        st.warning("Ningun remitente cumple el filtro seleccionado.")
        return

    # Se ordena por probabilidad descendente: el analista atiende primero lo
    # mas riesgoso, igual que en una cola real de cumplimiento.
    pool = sorted(pool, key=lambda r: -r["prob_b"])
    etiquetas = [
        f"{r['sender_id']}  -  P(lavado)={r['prob_b']:.1%}{'  [ALERTA]' if r['pred'] else ''}"
        for r in pool
    ]
    elegido = st.sidebar.selectbox(
        f"Remitente ({len(pool)} disponibles)", range(len(pool)),
        format_func=lambda i: etiquetas[i],
    )
    registro = pool[elegido]

    st.sidebar.divider()
    st.sidebar.caption(
        f"Umbral Etapa A: {thr_a:.4f}\n\n"
        f"Umbral Etapa B: {thr_b:.4f}\n\n"
        f"Tasa de alertas objetivo: {metrics['alert_rate']:.1%}\n\n"
        f"Longitud maxima de secuencia: {preprocessing['max_len']}"
    )

    # ---- Scores de ambas etapas --------------------------------------------
    df_seq = tabla_secuencia(registro)
    alerta = registro["pred"] == 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Etapa A - score de anomalia", f"{registro['anomaly_score_a']:.3f}",
        delta=f"{registro['anomaly_score_a'] - thr_a:+.3f} vs umbral",
        delta_color="inverse",
    )
    c2.metric(
        "Etapa B - P(lavado)", f"{registro['prob_b']:.1%}",
        delta=f"{(registro['prob_b'] - thr_b) * 100:+.1f} pp vs umbral",
        delta_color="inverse",
    )
    c3.metric("Decision del sistema", "ALERTA" if alerta else "Sin alerta")
    c4.metric("Etiqueta real", "Lavado" if registro["label"] == 1 else "Normal")

    if alerta:
        st.error(f"Caso alertado - tipologia detectada: **{registro['tipologia']}**")
    else:
        st.success(f"Caso sin alerta - tipologia detectada: **{registro['tipologia']}**")

    # ---- Mapa de calor sobre la secuencia ----------------------------------
    st.subheader("Mapa de calor de atencion sobre la secuencia")
    st.caption(
        "Cada celda es una transaccion; el color es su peso alpha en la capa de "
        "atencion. Los pesos suman 1 sobre la secuencia, por lo que representan "
        "el reparto de la evidencia que sustenta el score."
    )

    heat = (
        alt.Chart(df_seq)
        .mark_rect(stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("#:O", title="Indice de transaccion"),
            color=alt.Color(
                "Atencion:Q", title="alpha",
                scale=alt.Scale(scheme="inferno"),
            ),
            tooltip=[
                alt.Tooltip("#:O", title="Transaccion"),
                alt.Tooltip("Monto (USD):Q", format=",.2f"),
                "Tipo:N", "Hora:O", "Destino nuevo:N",
                alt.Tooltip("Atencion:Q", format=".4f"),
            ],
        )
        .properties(height=90)
    )
    barras = (
        alt.Chart(df_seq)
        .mark_bar(color="#c1121f")
        .encode(
            x=alt.X("#:O", title="Indice de transaccion"),
            y=alt.Y("Atencion:Q", title="alpha"),
            tooltip=[alt.Tooltip("Atencion:Q", format=".4f")],
        )
        .properties(height=160)
    )
    st.altair_chart(heat & barras, width="stretch")

    # ---- Secuencia de transacciones ----------------------------------------
    st.subheader("Secuencia de transacciones del remitente")
    st.dataframe(
        df_seq.style.background_gradient(subset=["Atencion"], cmap="inferno")
        .format({"Monto (USD)": "{:,.2f}", "Atencion": "{:.4f}"}),
        width="stretch", hide_index=True,
    )

    # ---- Explicacion en lenguaje natural -----------------------------------
    st.subheader("Explicacion automatica de la alerta")
    st.info(generar_explicacion(registro, df_seq, thr_a, thr_b))


if __name__ == "__main__":
    main()
