import random


def read_articles_categories():
    articles_categories = [(open(item.split()[1], encoding='utf-8').read(), item.split()[0])
                           for item in open('list.txt').read().strip().split('\n')]
    random.shuffle(articles_categories)
    return articles_categories


class ClassificationTester:
    def __init__(self, classifier, vectorizer):
        self.classifier = classifier
        self.vectorizer = vectorizer

    def train(self, articles_categories):
        vector = self.vectorizer.fit_transform(a for a, c in articles_categories)
        try:
            self.classifier.fit(vector, [c for a, c in articles_categories])
        except TypeError as e:
            if str(e) == 'A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.':
                self.classifier.fit(vector.toarray(), [c for a, c in articles_categories])
            else:
                raise

    def classify_many(self, articles):
        vector = self.vectorizer.transform(articles)
        try:
            return self.classifier.predict(vector)
        except TypeError as e:
            if str(e) == 'A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.':
                return self.classifier.predict(vector.toarray())
            else:
                raise

    def accuracy(self, articles_categories):
        articles_categories = list(articles_categories)
        articles = [a for a, c in articles_categories]
        categories = [c for a, c in articles_categories]
        predicted = self.classify_many(articles)
        correct = sum(c == p for c, p in zip(categories, predicted))
        return correct / len(articles_categories)
