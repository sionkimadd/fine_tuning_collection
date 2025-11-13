from transformers import pipeline

pipe = pipeline("text-classification", model="logicalqubit/deberta-v3-large-news-classifier")

texts = [
    "Major nations agree on first-ever global fee on greenhouse gases with plan that targets shipping",
    "US consumer sentiment plummets to second-lowest level on records going back to 1952",
    "DARPA eyes companies targeting industrially useful quantum computers",
    "Queen legend Brian May officially introduces Benson Boone as his new friend",
    "Maple Leafs will be down a player against the Montreal Canadiens on Saturday",
    "McGill discovery sheds new light on autism, intellectual disabilities",
    "Coca-Cola Coliseum concertgoers exposed to measles: Toronto Public Health",
]

for text in texts:
    result = pipe(text)[0]
    print(f"Text: {text}")
    print(f"Category: {result['label']} (Confidence: {result['score']:.4f})\n")