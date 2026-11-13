"""
utilidades.py
-------------
Funciones de apoyo: carga y filtrado del dataset, constantes compartidas
y conversión de estrellas a etiquetas de sentimiento.

Todas las rutas son relativas a la raíz del repositorio (se calculan a partir
de la ubicación de este archivo), por lo que funcionan igual en Colab,
en local y en Streamlit Cloud.
"""
from pathlib import Path

import pandas as pd

# --- Rutas relativas a la raíz del repositorio ------------------------------
RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "resenas_cusco.csv"
RUTA_MODELOS = RAIZ / "models"

# --- Constantes de sentimiento ----------------------------------------------
ETIQUETAS = ["NEG", "NEU", "POS"]                      # orden fijo para métricas
NOMBRES_SENTIMIENTO = {"NEG": "Negativo", "NEU": "Neutro", "POS": "Positivo"}
COLORES_SENTIMIENTO = {"Negativo": "#d62728", "Neutro": "#999999", "Positivo": "#2ca02c"}


def estrellas_a_sentimiento(estrellas: int) -> str:
    """Convierte la puntuación de Google (1-5) en una etiqueta de sentimiento.

    Regla (convención habitual en la literatura de reseñas):
        1-2 estrellas -> NEG,  3 estrellas -> NEU,  4-5 estrellas -> POS
    """
    if estrellas <= 2:
        return "NEG"
    if estrellas == 3:
        return "NEU"
    return "POS"


def cargar_resenas(ruta=None) -> pd.DataFrame:
    """Lee el CSV crudo tal como lo entrega el scraper de Google Maps."""
    return pd.read_csv(ruta or RUTA_DATOS)


def filtrar_resenas_espanol(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Deja solo las reseñas útiles para el análisis en español.

    Pasos (cada uno justificado en el notebook):
      1. Elimina reseñas sin texto (el usuario solo puso estrellas).
      2. Conserva únicamente `originalLanguage == "es"`: los modelos usados
         (robertuito, spaCy es_core_news_sm, stopwords) son para español.
      3. Elimina duplicados exactos (mismo alojamiento y mismo texto).
      4. Convierte la fecha y crea columnas auxiliares.
    """
    n0 = len(df)
    d = df.copy()
    d = d[d["text"].notna() & (d["text"].astype(str).str.strip() != "")]
    n1 = len(d)
    d = d[d["originalLanguage"] == "es"]
    n2 = len(d)
    d = d.drop_duplicates(subset=["title", "text"])
    n3 = len(d)

    d["fecha"] = pd.to_datetime(d["publishedAtDate"], errors="coerce", utc=True)
    d["n_palabras"] = d["text"].str.split().str.len()
    d["sent_estrellas"] = d["stars"].apply(estrellas_a_sentimiento)
    d = d.reset_index(drop=True)

    if verbose:
        print(f"Filas originales ............ {n0}")
        print(f"Con texto ................... {n1}")
        print(f"En español .................. {n2}")
        print(f"Sin duplicados exactos ...... {n3}")
    return d
