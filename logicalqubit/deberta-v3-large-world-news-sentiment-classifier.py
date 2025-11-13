from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="logicalqubit/deberta-v3-large-world-news-sentiment-classifier",
    tokenizer="logicalqubit/deberta-v3-large-world-news-sentiment-classifier",
    top_k=None,
    device=-1
)

texts = [
    "18 people sent to the hospital after mobile lounge crashes at Washington D.C.-area airport",
]

results = classifier(texts)

for text, scores in zip(texts, results):
    print(f"\n>>> {text}")
    for s in scores:
        print(f"  {s['label']:>8} : {s['score']:.4f}")