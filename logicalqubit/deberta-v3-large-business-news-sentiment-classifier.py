from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="logicalqubit/deberta-v3-large-business-news-sentiment-classifier",
    tokenizer="logicalqubit/deberta-v3-large-business-news-sentiment-classifier",
    top_k=None,
    device=-1
)

texts = [
    "AMD surges after event highlighting AI-driven growth keeps analysts bullish",
]

results = classifier(texts)

for text, scores in zip(texts, results):
    print(f"\n>>> {text}")
    for s in scores:
        print(f"  {s['label']:>8} : {s['score']:.4f}")