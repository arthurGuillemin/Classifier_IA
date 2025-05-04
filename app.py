import gradio as gr
from transformers import pipeline
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import pandas as pd

# DATA AUGMENTATION 
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

MODEL_ID = "tribber93/my-trash-classification"
trash_classifier = pipeline(
    "image-classification",
    model=MODEL_ID,
    device=0 if torch.cuda.is_available() else -1,
    top_k=3
)


# MAPPING
POUBELLES = {
    "cardboard": "papier/carton",
    "glass": "verre",
    "metal": "métal",
    "paper": "papier",
    "plastic": "plastique",
    "trash": "ordures ménagères",
}

#CLASSIFICATION
def classify_image(image: Image.Image):
    image_aug = augment(image)
    results = trash_classifier(image_aug)

    rows = []
    for r in results:
        label = r["label"]
        score = r["score"]
        poubelle = POUBELLES.get(label.lower(), "inconnue")
        rows.append({
            "Objet": label,
            "Poubelle": poubelle,
            "Confiance (%)": round(score * 100, 2)
        })
    return pd.DataFrame(rows)

#GRADIO
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Dataframe(
        headers=["Objet", "Poubelle", "Confiance (%)"],
        row_count=(1, 10)
    ),
    title="🗑️ Trash classifier 🗑️ ",
    description=(
        "Dépose une image de déchet pour savoir dans quelle poubelle la trier "
        "Le modèle est fine-tuné sur TrashNet et bénéficie de data augmentation pour une meilleure robustesse."
    ),
    examples=None,
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()
