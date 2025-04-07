# model_wrapper.py

class SentimentModel:
    def __init__(self, model, vectorizer, label_map_rev):
        self.model = model
        self.vectorizer = vectorizer
        self.label_map_rev = label_map_rev

    def predict(self, texts):
        tfidf = self.vectorizer.transform(texts)
        preds = self.model.predict(tfidf)
        return [self.label_map_rev[p] for p in preds]
