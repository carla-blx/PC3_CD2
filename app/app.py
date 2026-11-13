"""
Aplicación Streamlit: análisis de reseñas de alojamientos de Cusco.

Funciones:
  * Entrada: texto manual (una reseña por línea) o archivo CSV.
  * Salida: sentimiento, palabras clave (TF-IDF), tema (LDA), tablas y gráficos interactivos.

Ejecutar en local desde la raíz del repo:   streamlit run app/app.py
"""
import sys
from pathlib import Path

# Permite importar el paquete `src` sin importar desde dónde se lance la app.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.modelos import (asignar_temas, palabras_clave_por_documento, predecir_sentimiento_ligero,
                         top_palabras_tema, top_terminos)
from src.preprocesamiento import cargar_spacy, preprocesar_corpus
from src.utilidades import COLORES_SENTIMIENTO, NOMBRES_SENTIMIENTO

st.set_page_config(page_title="Reseñas de alojamientos en Cusco", page_icon="🏨", layout="wide")

MAX_FILAS = 3000          # tope para que la app responda rápido en Streamlit Cloud
SIN_TEMA = "Sin tema (texto muy corto)"
ORDEN_SENT = list(COLORES_SENTIMIENTO)   # ["Negativo", "Neutro", "Positivo"]


# ---------------------------------------------------------------------------
# Carga de recursos (una sola vez por proceso del servidor)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando spaCy y modelos (la primera vez tarda un poco)...")
def cargar_recursos():
    from src.modelos import cargar_o_entrenar
    nlp = cargar_spacy()
    return nlp, cargar_o_entrenar(nlp)


def cargar_transformer():
    from src.modelos import cargar_analizador_transformer
    return cargar_analizador_transformer()


nlp, ART = cargar_recursos()


# ---------------------------------------------------------------------------
# Pipeline completo sobre una lista de textos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Procesando textos...")
def analizar(textos: tuple, n_claves: int, usar_transformer: bool) -> pd.DataFrame:
    limpios, lemas_tfidf, lemas_lda = preprocesar_corpus(list(textos), nlp)

    # 1) Sentimiento
    if usar_transformer:
        sent = cargar_transformer()(list(textos))
    else:
        sent = predecir_sentimiento_ligero(ART["sentimiento"], limpios)

    # 2) Palabras clave con TF-IDF
    claves = palabras_clave_por_documento(ART["tfidf"].transform(lemas_tfidf), ART["tfidf"], n=n_claves)

    # 3) Tema con LDA (reseñas con <3 lemas no tienen información suficiente)
    temas, prob, _ = asignar_temas(ART["cv"], ART["lda"], lemas_lda)
    corto = np.array([len(d.split()) < 3 for d in lemas_lda])
    etiquetas = np.array(list(ART["etiquetas_temas"]) + [SIN_TEMA], dtype=object)
    nombre_tema = np.where(corto | (temas < 0), SIN_TEMA, etiquetas[np.where(temas < 0, -1, temas)])

    return pd.DataFrame({
        "texto": list(textos),
        "sentimiento": sent["pred"].map(NOMBRES_SENTIMIENTO).values,
        "prob_positivo": sent["POS"].astype(float).round(3).values,
        "prob_negativo": sent["NEG"].astype(float).round(3).values,
        "palabras_clave": claves,
        "tema": nombre_tema,
        "prob_tema": np.round(prob, 3),
        "lemas": lemas_tfidf,
    })


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Opciones")
modo = st.sidebar.radio("Entrada de datos", ["Texto manual", "Subir CSV"])
n_claves = st.sidebar.slider("Palabras clave por reseña", 3, 10, 5)

usar_transformer = False

st.sidebar.caption("Modelo por defecto: TF-IDF + Regresión Logística entrenada con las estrellas.")

# ---------------------------------------------------------------------------
# Cabecera y entrada de datos
# ---------------------------------------------------------------------------
st.title("🏨 Análisis de reseñas de alojamientos en Cusco")
st.write("Sentimiento, palabras clave (TF-IDF) y temas (LDA) sobre reseñas en español de Google Maps.")

df_in = None
if modo == "Texto manual":
    ejemplo = ("Excelente atención del personal, la habitación muy limpia y cerca de la plaza de armas.\n"
               "No había agua caliente y el desayuno era pésimo, no lo recomiendo.")
    texto = st.text_area("Escribe una reseña por línea:", value=ejemplo, height=160)
    textos = [t.strip() for t in texto.splitlines() if t.strip()]
    if textos:
        df_in = pd.DataFrame({"text": textos})
else:
    archivo = st.file_uploader("Sube un CSV con reseñas", type=["csv"])
    if archivo is not None:
        try:
            df_in = pd.read_csv(archivo)
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
            df_in = None
        if df_in is not None:
            candidatas = [c for c in df_in.columns if not pd.api.types.is_numeric_dtype(df_in[c])]
            if not candidatas:
                st.error("El CSV no tiene columnas de texto.")
                df_in = None
            else:
                por_defecto = candidatas.index("text") if "text" in candidatas else 0
                col = st.selectbox("Columna con el texto de la reseña", candidatas, index=por_defecto)
                df_in = df_in[df_in[col].notna() & (df_in[col].astype(str).str.strip() != "")].copy()
                df_in["text"] = df_in[col].astype(str)
                if "originalLanguage" in df_in and st.checkbox(
                        "Solo reseñas en español (originalLanguage = es)", True):
                    df_in = df_in[df_in["originalLanguage"] == "es"]
                if len(df_in) > MAX_FILAS:
                    st.info(f"Se analizarán las primeras {MAX_FILAS} de {len(df_in)} reseñas.")
                    df_in = df_in.head(MAX_FILAS)
                df_in = df_in.reset_index(drop=True)

if df_in is None or df_in.empty:
    st.info("Ingresa texto o sube un CSV para comenzar.")
    st.stop()

res = analizar(tuple(df_in["text"]), n_claves, usar_transformer)
if "stars" in df_in:
    res.insert(0, "estrellas", df_in["stars"].values)
if "title" in df_in:
    res.insert(0, "alojamiento", df_in["title"].values)

# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Reseñas analizadas", len(res))
c2.metric("% positivas", f"{(res['sentimiento'] == 'Positivo').mean() * 100:.1f}%")
c3.metric("% negativas", f"{(res['sentimiento'] == 'Negativo').mean() * 100:.1f}%")

tab_res, tab_tabla, tab_temas, tab_claves = st.tabs(
    ["📊 Resumen", "📋 Tabla", "🧩 Temas (LDA)", "🔑 Palabras clave (TF-IDF)"])

with tab_res:
    a, b = st.columns(2)
    cuenta = res["sentimiento"].value_counts().reindex(ORDEN_SENT).fillna(0).reset_index()
    cuenta.columns = ["sentimiento", "reseñas"]
    a.plotly_chart(px.bar(cuenta, x="sentimiento", y="reseñas", color="sentimiento",
                          color_discrete_map=COLORES_SENTIMIENTO, title="Distribución de sentimiento"),
                   width="stretch")
    por_tema = res.groupby(["tema", "sentimiento"]).size().reset_index(name="reseñas")
    b.plotly_chart(px.bar(por_tema, x="tema", y="reseñas", color="sentimiento",
                          color_discrete_map=COLORES_SENTIMIENTO, title="Sentimiento por tema"),
                   width="stretch")

    if "estrellas" in res:
        st.subheader("Sentimiento predicho vs. estrellas")
        real = res["estrellas"].map(lambda s: "Negativo" if s <= 2 else "Neutro" if s == 3 else "Positivo")
        matriz = pd.crosstab(real, res["sentimiento"]).reindex(index=ORDEN_SENT, columns=ORDEN_SENT, fill_value=0)
        st.plotly_chart(px.imshow(matriz, text_auto=True, color_continuous_scale="Blues",
                                  labels=dict(x="Predicho", y="Según estrellas", color="Reseñas")),
                        width="stretch")

with tab_tabla:
    filtro = st.multiselect("Filtrar por sentimiento", ORDEN_SENT, default=ORDEN_SENT)
    vista = res[res["sentimiento"].isin(filtro)].drop(columns=["lemas"])
    st.dataframe(vista, width="stretch", height=420)
    st.download_button("⬇️ Descargar resultados (CSV)", vista.to_csv(index=False).encode("utf-8"),
                       file_name="resultados_resenas.csv", mime="text/csv")

with tab_temas:
    st.write("Cada tema es una mezcla de palabras que suelen aparecer juntas. "
             "El nombre provisional se construye con sus palabras más probables.")
    palabras = top_palabras_tema(ART["lda"], ART["cv"], n=10)
    for etiqueta, pals in zip(ART["etiquetas_temas"], palabras):
        st.markdown(f"**{etiqueta}** — " + ", ".join(pals))
    conteo = res["tema"].value_counts().reset_index()
    conteo.columns = ["tema", "reseñas"]
    st.plotly_chart(px.bar(conteo, x="reseñas", y="tema", orientation="h", title="Reseñas por tema"),
                    width="stretch")

with tab_claves:
    X = ART["tfidf"].transform(res["lemas"])
    n_top = st.slider("Cantidad de términos", 5, 30, 15)
    tipo = st.radio("Tipo", ["Todos", "Solo unigramas", "Solo bigramas"], horizontal=True)
    top = top_terminos(X, ART["tfidf"], n=n_top, solo_unigramas=(tipo == "Solo unigramas"),
                       solo_bigramas=(tipo == "Solo bigramas"))
    st.plotly_chart(px.bar(top.iloc[::-1], x="tfidf_medio", y="termino", orientation="h",
                           title="Términos con mayor TF-IDF promedio en tus reseñas"),
                    width="stretch")
