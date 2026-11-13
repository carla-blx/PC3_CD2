"""
modelos.py
----------
Modelos y técnicas de PLN del proyecto:

  1. Sentimiento preentrenado en español (robertuito, vía pysentimiento o transformers).
  2. Sentimiento ligero: TF-IDF + Regresión Logística entrenada con las estrellas
     (para la app en Streamlit Cloud, que tiene poca memoria).
  3. TF-IDF para palabras clave (unigramas y bigramas).
  4. LDA (topic modeling) con scikit-learn.
  5. Guardado / carga de artefactos con joblib y reentrenamiento de respaldo.
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .utilidades import ETIQUETAS, RUTA_MODELOS, estrellas_a_sentimiento

SEMILLA = 42
MODELO_TRANSFORMER = "pysentimiento/robertuito-sentiment-analysis"
# robertuito acepta ~128 tokens; recortamos el texto de entrada para evitar errores.
TRUNCAR_CARACTERES = 600


# ===========================================================================
# 1. SENTIMIENTO PREENTRENADO (robertuito)
# ===========================================================================
def cargar_analizador_transformer():
    """Devuelve una función `predecir(textos) -> DataFrame[pred, NEG, NEU, POS]`.

    Intenta primero con `pysentimiento` (librería oficial del modelo, que además
    aplica su propio preprocesado). Si no se puede importar (por conflicto de
    versiones), usa `transformers` directamente con el mismo modelo.
    """
    try:
        from pysentimiento import create_analyzer

        analizador = create_analyzer(task="sentiment", lang="es")

        def predecir(textos, tam_lote: int = 64):
            filas = []
            textos = [str(t)[:TRUNCAR_CARACTERES] for t in textos]
            for i in range(0, len(textos), tam_lote):
                for r in analizador.predict(textos[i:i + tam_lote]):
                    filas.append({"pred": r.output, **{k: r.probas.get(k, 0.0) for k in ETIQUETAS}})
            return pd.DataFrame(filas)

        return predecir
    except Exception as e:  # ImportError u otro conflicto de versiones
        warnings.warn(f"pysentimiento no disponible ({e}); se usa transformers directamente.")

    from transformers import pipeline

    clf = pipeline("text-classification", model=MODELO_TRANSFORMER, tokenizer=MODELO_TRANSFORMER,
                   top_k=None, truncation=True, max_length=128)

    def predecir(textos, tam_lote: int = 32):
        filas = []
        textos = [str(t)[:TRUNCAR_CARACTERES] for t in textos]
        for i in range(0, len(textos), tam_lote):
            for salida in clf(textos[i:i + tam_lote]):
                # Las etiquetas del modelo son NEG / NEU / POS
                probs = {d["label"][:3].upper(): d["score"] for d in salida}
                fila = {k: probs.get(k, 0.0) for k in ETIQUETAS}
                filas.append({"pred": max(fila, key=fila.get), **fila})
        return pd.DataFrame(filas)

    return predecir


# ===========================================================================
# 2. SENTIMIENTO LIGERO (TF-IDF + Regresión Logística)
# ===========================================================================
def entrenar_sentimiento_ligero(textos_limpios, etiquetas, test_size=0.2, semilla=SEMILLA):
    """Entrena un clasificador NEG/NEU/POS usando las estrellas como etiqueta.

    Devuelve el pipeline entrenado, los índices del split y las métricas en test.
    `class_weight="balanced"` compensa que ~2/3 de las reseñas son de 5 estrellas.
    """
    textos_limpios = np.asarray(textos_limpios, dtype=object)
    etiquetas = np.asarray(etiquetas, dtype=object)
    idx = np.arange(len(textos_limpios))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, random_state=semilla, stratify=etiquetas)

    modelo = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    modelo.fit(textos_limpios[idx_tr], etiquetas[idx_tr])
    pred = modelo.predict(textos_limpios[idx_te])
    metricas = {
        "accuracy": float(accuracy_score(etiquetas[idx_te], pred)),
        "macro_f1": float(f1_score(etiquetas[idx_te], pred, average="macro", labels=ETIQUETAS)),
        "n_train": int(len(idx_tr)),
        "n_test": int(len(idx_te)),
    }
    return modelo, idx_tr, idx_te, metricas


def predecir_sentimiento_ligero(modelo, textos_limpios) -> pd.DataFrame:
    """Devuelve DataFrame con columnas pred, NEG, NEU, POS (probabilidades)."""
    if len(textos_limpios) == 0:
        return pd.DataFrame(columns=["pred"] + ETIQUETAS)
    probs = modelo.predict_proba(textos_limpios)
    df = pd.DataFrame(probs, columns=modelo.classes_)
    for e in ETIQUETAS:
        if e not in df:
            df[e] = 0.0
    df["pred"] = modelo.predict(textos_limpios)
    return df[["pred"] + ETIQUETAS]


# ===========================================================================
# 3. TF-IDF: términos y bigramas relevantes
# ===========================================================================
def entrenar_tfidf(lemas, ngram_range=(1, 2), min_df=3, max_df=0.8):
    """Ajusta un TfidfVectorizer sobre los lemas.

    TF-IDF(t, d) = TF(t, d) * IDF(t)  con  IDF(t) = ln((1+N)/(1+df(t))) + 1 (versión suavizada de sklearn).
    `max_df` descarta términos presentes en >80 % de documentos; `min_df` descarta los muy raros.
    """
    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df, sublinear_tf=True)
    X = vec.fit_transform(lemas)
    return vec, X


def top_terminos(X, vec, n=15, solo_bigramas=False, solo_unigramas=False) -> pd.DataFrame:
    """Términos con mayor peso TF-IDF promedio en el corpus."""
    nombres = np.array(vec.get_feature_names_out())
    media = np.asarray(X.mean(axis=0)).ravel()
    es_bi = np.array([" " in t for t in nombres])
    mascara = np.ones(len(nombres), dtype=bool)
    if solo_bigramas:
        mascara = es_bi
    if solo_unigramas:
        mascara = ~es_bi
    orden = np.argsort(media)[::-1]
    orden = [i for i in orden if mascara[i]][:n]
    return pd.DataFrame({"termino": nombres[orden], "tfidf_medio": media[orden]})


def palabras_clave_por_documento(X, vec, n=5):
    """Para cada documento, los n términos con mayor TF-IDF (texto separado por comas)."""
    nombres = np.array(vec.get_feature_names_out())
    claves = []
    for i in range(X.shape[0]):
        fila = X.getrow(i)
        if fila.nnz == 0:
            claves.append("")
            continue
        orden = np.argsort(fila.data)[::-1][:n]
        claves.append(", ".join(nombres[fila.indices[orden]]))
    return claves


# ===========================================================================
# 4. LDA (topic modeling)
# ===========================================================================
def entrenar_lda(lemas, k=5, semilla=SEMILLA, min_df=5, max_df=0.5, max_iter=30):
    """Ajusta CountVectorizer + LDA. LDA trabaja con CONTEOS (bolsa de palabras), no con TF-IDF."""
    cv = CountVectorizer(min_df=min_df, max_df=max_df)
    X = cv.fit_transform(lemas)
    lda = LatentDirichletAllocation(n_components=k, learning_method="batch", max_iter=max_iter,
                                    random_state=semilla)
    lda.fit(X)
    return cv, lda, X


def top_palabras_tema(lda, cv, n=10):
    """Lista (una por tema) con las n palabras de mayor peso."""
    nombres = np.array(cv.get_feature_names_out())
    return [list(nombres[np.argsort(comp)[::-1][:n]]) for comp in lda.components_]


def coherencia_umass(lda, X, n=10):
    """Coherencia UMass: mide si las palabras top de un tema co-ocurren en los mismos documentos.

    Valores más cercanos a 0 (menos negativos) = temas más coherentes.
    C = media sobre pares (wi, wj) de log((D(wi,wj)+1) / D(wj))
    """
    binaria = (X > 0).astype(int).tocsc()
    coocurrencia = (binaria.T @ binaria).toarray()
    doc_freq = np.diag(coocurrencia)
    puntajes = []
    for comp in lda.components_:
        top = np.argsort(comp)[::-1][:n]
        s = 0.0
        for a in range(1, len(top)):
            for b in range(a):
                s += np.log((coocurrencia[top[a], top[b]] + 1) / doc_freq[top[b]])
        puntajes.append(s / (n * (n - 1) / 2))
    return float(np.mean(puntajes))


def asignar_temas(cv, lda, lemas):
    """Devuelve (tema dominante, probabilidad del dominante, matriz doc-tema).

    Documentos sin ninguna palabra del vocabulario reciben tema -1 (sin tema).
    """
    X = cv.transform(lemas)
    dist = lda.transform(X)
    sin_vocab = np.asarray(X.sum(axis=1)).ravel() == 0
    tema = dist.argmax(axis=1)
    prob = dist.max(axis=1)
    tema = np.where(sin_vocab, -1, tema)
    prob = np.where(sin_vocab, np.nan, prob)
    return tema, prob, dist


def etiquetas_automaticas(palabras_por_tema, n=3):
    """Nombre provisional de cada tema con sus n palabras principales."""
    return [f"Tema {i + 1}: " + ", ".join(p[:n]) for i, p in enumerate(palabras_por_tema)]


# ===========================================================================
# 5. ARTEFACTOS
# ===========================================================================
def guardar_artefactos(sent_modelo, tfidf_vec, cv, lda, etiquetas_temas, metricas, carpeta=RUTA_MODELOS):
    carpeta.mkdir(parents=True, exist_ok=True)
    joblib.dump(sent_modelo, carpeta / "sentimiento_ligero.joblib", compress=3)
    joblib.dump(tfidf_vec, carpeta / "tfidf_palabras_clave.joblib", compress=3)
    joblib.dump({"cv": cv, "lda": lda}, carpeta / "lda_temas.joblib", compress=3)
    meta = {"sklearn_version": sklearn.__version__, "k_temas": int(lda.n_components),
            "etiquetas_temas": list(etiquetas_temas), "metricas_sentimiento_ligero": metricas}
    (carpeta / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def cargar_artefactos(carpeta=RUTA_MODELOS):
    """Carga los modelos guardados. Devuelve None si faltan o si la versión de scikit-learn difiere
    (los .joblib no son portables entre versiones; en ese caso conviene reentrenar)."""
    try:
        meta = json.loads((carpeta / "metadata.json").read_text(encoding="utf-8"))
        if meta.get("sklearn_version") != sklearn.__version__:
            return None
        lda_dic = joblib.load(carpeta / "lda_temas.joblib")
        etiquetas = meta["etiquetas_temas"]
        manual = carpeta / "etiquetas_temas.json"       # opcional: nombres escritos a mano
        if manual.exists():
            lista = json.loads(manual.read_text(encoding="utf-8"))
            if len(lista) == len(etiquetas):
                etiquetas = lista
        return {"sentimiento": joblib.load(carpeta / "sentimiento_ligero.joblib"),
                "tfidf": joblib.load(carpeta / "tfidf_palabras_clave.joblib"),
                "cv": lda_dic["cv"], "lda": lda_dic["lda"],
                "etiquetas_temas": etiquetas, "metricas": meta.get("metricas_sentimiento_ligero", {})}
    except Exception:
        return None


def entrenar_todo(df, nlp, k=5):
    """Entrena todos los modelos ligeros a partir del DataFrame filtrado (columnas text, stars)."""
    from .preprocesamiento import preprocesar_corpus

    limpios, lemas_tfidf, lemas_lda = preprocesar_corpus(df["text"].tolist(), nlp)
    sent, _, _, metricas = entrenar_sentimiento_ligero(limpios, df["stars"].apply(estrellas_a_sentimiento))
    tfidf_vec, _ = entrenar_tfidf(lemas_tfidf)
    docs_lda = [d for d in lemas_lda if len(d.split()) >= 3]      # descarta reseñas casi vacías
    cv, lda, _ = entrenar_lda(docs_lda, k=k)
    etiquetas = etiquetas_automaticas(top_palabras_tema(lda, cv))
    return {"sentimiento": sent, "tfidf": tfidf_vec, "cv": cv, "lda": lda,
            "etiquetas_temas": etiquetas, "metricas": metricas}


def cargar_o_entrenar(nlp, k=5):
    """Usa los artefactos guardados; si no son utilizables, reentrena desde data/ (respaldo)."""
    art = cargar_artefactos()
    if art is not None:
        return art
    from .utilidades import cargar_resenas, filtrar_resenas_espanol
    return entrenar_todo(filtrar_resenas_espanol(cargar_resenas()), nlp, k=k)
