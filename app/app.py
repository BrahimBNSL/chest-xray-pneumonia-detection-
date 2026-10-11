"""Interface web Gradio : envoi d'une radiographie et affichage de la prédiction."""

from __future__ import annotations

import gradio as gr
from PIL import Image

from app.predictor import DISPLAY_NAMES, InvalidImageError, Predictor
from src.config import CLASS_NAMES

AVERTISSEMENT: str = """
**Outil d'aide au tri, pas un diagnostic.**
Ce modèle a été entraîné sur des radiographies d'enfants de 1 à 5 ans. Son résultat ne remplace ni l'avis d'un radiologue, ni un examen clinique. Ne prenez aucune décision médicale sur la seule base de cette prédiction.
Les images envoyées ne sont pas conservées par l'application.
"""

_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    """Charge le modèle à la première demande, puis le réutilise."""
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def analyser(image: Image.Image | None):
    """Analyse une image et retourne les probabilités et un résumé textuel."""
    if image is None:
        return {}, "Chargez une radiographie pour lancer l'analyse."

    try:
        resultat = get_predictor().predict(image)
    except InvalidImageError as erreur:
        return {}, f"**Image refusée.** {erreur}"

    probabilites = {
        DISPLAY_NAMES[CLASS_NAMES[0]]: 1 - resultat.probabilite_pneumonie,
        DISPLAY_NAMES[CLASS_NAMES[1]]: resultat.probabilite_pneumonie,
    }
    resume = (
        f"### Résultat : {resultat.diagnostic}\n"
        f"Confiance du modèle : **{resultat.confiance * 100:.1f} %**\n\n"
        f"Probabilité de pneumonie : {resultat.probabilite_pneumonie * 100:.1f} %\n\n"
        "Rappel : ce résultat doit être confirmé par un professionnel de santé."
    )
    return probabilites, resume


def build_interface() -> gr.Blocks:
    """Construit l'interface : avertissement, zone d'envoi, probabilités et résumé."""
    with gr.Blocks(title="Détection de pneumonie sur radiographie") as demo:
        gr.Markdown("# Détection de pneumonie sur radiographie thoracique")
        gr.Markdown(AVERTISSEMENT)
        with gr.Row():
            entree = gr.Image(type="pil", label="Radiographie thoracique (face)", height=420)
            with gr.Column():
                probabilites = gr.Label(label="Probabilités", num_top_classes=2)
                resume = gr.Markdown()
        bouton = gr.Button("Analyser", variant="primary")
        bouton.click(fn=analyser, inputs=entree, outputs=[probabilites, resume])
    return demo


if __name__ == "__main__":
    build_interface().launch()