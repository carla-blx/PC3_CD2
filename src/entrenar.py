"""
Script para (re)generar los artefactos de models/ desde la línea de comandos:

    python -m src.entrenar            # k=5 temas por defecto
    python -m src.entrenar --k 6
"""
import argparse

from .modelos import entrenar_todo, guardar_artefactos
from .preprocesamiento import cargar_spacy
from .utilidades import cargar_resenas, filtrar_resenas_espanol

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="número de temas LDA")
    args = ap.parse_args()

    df = filtrar_resenas_espanol(cargar_resenas(), verbose=True)
    art = entrenar_todo(df, cargar_spacy(), k=args.k)
    guardar_artefactos(art["sentimiento"], art["tfidf"], art["cv"], art["lda"],
                       art["etiquetas_temas"], art["metricas"])
    print("Métricas del modelo ligero (test):", art["metricas"])
    print("Temas:", *art["etiquetas_temas"], sep="\n  ")
