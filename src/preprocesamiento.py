"""
preprocesamiento.py
-------------------
Pipeline de preprocesamiento de texto en español:

    texto crudo --> limpieza (minúsculas + regex + normalización)
                --> tokenización y lematización (spaCy es_core_news_sm)
                --> eliminación de stopwords
                --> texto de lemas listo para TF-IDF / LDA

Se separan dos niveles de limpieza a propósito:
  * `limpiar_texto`  -> conserva la frase casi completa (para el modelo de
                        sentimiento, que necesita el contexto y las negaciones).
  * `filtrar_lemas`  -> deja solo lemas con contenido (para TF-IDF y LDA).
"""
import re

NOMBRE_MODELO_SPACY = "es_core_news_sm"

# ---------------------------------------------------------------------------
# 1. LIMPIEZA CON EXPRESIONES REGULARES
# ---------------------------------------------------------------------------
RE_URL = re.compile(r"(https?://\S+|www\.\S+)")
RE_EMAIL = re.compile(r"\S+@\S+")
RE_HTML = re.compile(r"<[^>]+>")
# Todo lo que NO sea letra (incluye ñ y vocales con tilde/diéresis) se vuelve espacio.
# Esto elimina emojis, números, signos de puntuación y símbolos.
RE_NO_LETRAS = re.compile(r"[^a-záéíóúüñ\s]")
# Letras repetidas 3+ veces ("buenooo", "exceleeente") -> una sola.
RE_REPETIDAS = re.compile(r"([a-záéíóúüñ])\1{2,}")
RE_ESPACIOS = re.compile(r"\s+")

# Normalización de abreviaturas frecuentes en reseñas escritas desde el celular.
ABREVIATURAS = {
    "q": "que", "xq": "porque", "pq": "porque", "porq": "porque",
    "tb": "también", "tmb": "también", "tambien": "también",
    "bn": "bien", "x": "por", "k": "que",
}


def limpiar_texto(texto) -> str:
    """Minúsculas + regex + normalización de abreviaturas.

    Conserva las tildes (spaCy las necesita para lematizar bien).
    """
    if not isinstance(texto, str):
        return ""
    t = texto.lower()
    t = RE_URL.sub(" ", t)
    t = RE_EMAIL.sub(" ", t)
    t = RE_HTML.sub(" ", t)
    t = RE_NO_LETRAS.sub(" ", t)
    t = RE_REPETIDAS.sub(r"\1", t)
    palabras = [ABREVIATURAS.get(p, p) for p in t.split()]
    return RE_ESPACIOS.sub(" ", " ".join(palabras)).strip()


# ---------------------------------------------------------------------------
# 2. spaCy: TOKENIZACIÓN + LEMATIZACIÓN
# ---------------------------------------------------------------------------
def cargar_spacy(con_ner: bool = False):
    """Carga es_core_news_sm; si no está instalado intenta descargarlo.

    Se desactivan `parser` (y `ner` salvo que se pida) porque no los necesitamos
    para lematizar y así el procesamiento es bastante más rápido.
    """
    import spacy

    desactivar = ["parser"] + ([] if con_ner else ["ner"])
    try:
        return spacy.load(NOMBRE_MODELO_SPACY, disable=desactivar)
    except OSError:
        pass
    # Respaldo para entornos con permisos (local/Colab). En Streamlit Cloud NO funciona:
    # allí el modelo debe estar en requirements.txt.
    try:
        from spacy.cli import download
        download(NOMBRE_MODELO_SPACY)
        return spacy.load(NOMBRE_MODELO_SPACY, disable=desactivar)
    except (SystemExit, Exception) as e:
        raise RuntimeError(
            f"No se pudo cargar el modelo {NOMBRE_MODELO_SPACY}. Instálalo con "
            f"'python -m spacy download {NOMBRE_MODELO_SPACY}' o agrégalo a requirements.txt.") from e


# Negaciones: invierten el sentido ("no funciona", "sin agua"). spaCy las
# considera stopwords, así que las rescatamos cuando interesa conservarlas.
NEGACIONES = {"no", "nunca", "ni", "sin", "jamás", "tampoco", "nada"}

# Stopwords de dominio: aparecen en casi todas las reseñas y no discriminan.
STOPWORDS_DOMINIO = {
    "hotel", "hostal", "hostel", "cusco", "cuzco", "lugar", "hospedaje",
    "alojamiento", "albergue", "ser", "estar", "haber", "tener", "hacer",
    "ir", "poder", "dar", "ver", "decir", "querer", "así", "bien", "mas", "más",
}

# Categorías gramaticales con contenido temático (para LDA).
POS_CONTENIDO = {"NOUN", "PROPN", "ADJ", "VERB"}


def obtener_stopwords(nlp, conservar_negaciones: bool = False) -> set:
    """Stopwords en español de spaCy + stopwords de dominio (− negaciones si se pide)."""
    stop = set(nlp.Defaults.stop_words) | STOPWORDS_DOMINIO
    if conservar_negaciones:
        stop -= NEGACIONES
    return stop


def analizar_textos(textos, nlp, batch_size: int = 128):
    """Tokeniza y lematiza una lista de textos YA limpios.

    Devuelve, por cada texto, una lista de tuplas (lema, categoría_gramatical).
    Se hace UNA sola pasada por spaCy y luego `filtrar_lemas` produce las
    distintas variantes (TF-IDF vs LDA) sin volver a procesar.
    """
    resultado = []
    for doc in nlp.pipe(textos, batch_size=batch_size):
        resultado.append([(t.lemma_.lower(), t.pos_) for t in doc if t.is_alpha])
    return resultado


def filtrar_lemas(tokens, stopwords: set, pos_validos=None, min_len: int = 3) -> str:
    """Quita stopwords y tokens cortos; opcionalmente filtra por categoría gramatical.

    Ejemplo: [("limpio","ADJ"),("el","DET"),("habitación","NOUN")] -> "limpio habitación"
    """
    lemas = []
    for lema, pos in tokens:
        if lema in stopwords:
            continue
        if len(lema) < min_len and lema not in NEGACIONES:
            continue
        if pos_validos is not None and pos not in pos_validos:
            continue
        lemas.append(lema)
    return " ".join(lemas)


def preprocesar_corpus(textos_crudos, nlp):
    """Atajo que ejecuta todo el pipeline y devuelve tres listas paralelas.

    Returns
    -------
    limpios    : texto en minúsculas sin ruido (para sentimiento)
    lemas_tfidf: lemas sin stopwords, conservando negaciones (para TF-IDF)
    lemas_lda  : lemas de contenido (sustantivos/adjetivos/verbos) sin negaciones (para LDA)
    """
    limpios = [limpiar_texto(t) for t in textos_crudos]
    tokens = analizar_textos(limpios, nlp)
    stop_neg = obtener_stopwords(nlp, conservar_negaciones=True)
    stop_lda = obtener_stopwords(nlp, conservar_negaciones=False)
    lemas_tfidf = [filtrar_lemas(t, stop_neg) for t in tokens]
    lemas_lda = [filtrar_lemas(t, stop_lda, pos_validos=POS_CONTENIDO) for t in tokens]
    return limpios, lemas_tfidf, lemas_lda
