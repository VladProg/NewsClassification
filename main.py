from time import time
from classification_tester import read_articles_categories, ClassificationTester
import sklearn.calibration, sklearn.discriminant_analysis, sklearn.dummy, sklearn.ensemble, sklearn.gaussian_process, sklearn.linear_model, sklearn.multiclass, sklearn.naive_bayes, sklearn.neighbors, sklearn.neural_network, sklearn.svm, sklearn.tree
import sklearn.feature_extraction
from collections import Counter

articles_categories = read_articles_categories()
print('number of articles =', len(articles_categories), ':', dict(Counter(c for a, c in articles_categories)))
train_set_size = int(input('input size of train set: '))
assert 0 < train_set_size < len(articles_categories)
train_set, test_set = articles_categories[:train_set_size], articles_categories[train_set_size:]

classifiers = [
    sklearn.calibration.CalibratedClassifierCV,
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis,
    sklearn.dummy.DummyClassifier,
    sklearn.ensemble.AdaBoostClassifier,
    sklearn.ensemble.BaggingClassifier,
    sklearn.ensemble.ExtraTreesClassifier,
    sklearn.ensemble.GradientBoostingClassifier,
    sklearn.ensemble.RandomForestClassifier,
    sklearn.gaussian_process.GaussianProcessClassifier,
    sklearn.linear_model.LogisticRegression,
    sklearn.linear_model.LogisticRegressionCV,
    sklearn.linear_model.PassiveAggressiveClassifier,
    sklearn.linear_model.Perceptron,
    sklearn.linear_model.RidgeClassifier,
    sklearn.linear_model.RidgeClassifierCV,
    sklearn.linear_model.SGDClassifier,
    sklearn.naive_bayes.BernoulliNB,
    sklearn.naive_bayes.CategoricalNB,
    sklearn.naive_bayes.ComplementNB,
    sklearn.naive_bayes.GaussianNB,
    sklearn.naive_bayes.MultinomialNB,
    sklearn.neighbors.KNeighborsClassifier,
    sklearn.neighbors.NearestCentroid,
    sklearn.neural_network.MLPClassifier,
    sklearn.svm.LinearSVC,
    sklearn.svm.NuSVC,
    sklearn.svm.SVC,
    sklearn.tree.DecisionTreeClassifier,
    sklearn.tree.ExtraTreeClassifier
]

vectorizers = [
    sklearn.feature_extraction.text.CountVectorizer,
    sklearn.feature_extraction.text.HashingVectorizer,
    sklearn.feature_extraction.text.TfidfVectorizer
]

ngram_ranges = [(1, 1), (2, 2), (1, 2)]

with open(f'stats_{train_set_size}.csv', 'w') as stats, open(f'exceptions_{train_set_size}.txt', 'w') as exceptions:
    stats.write(''.join(f',"{vectorizer.__name__}(ngram_range={ngram_range})"'
                        for vectorizer in vectorizers
                        for ngram_range in ngram_ranges) + '\n')
    for classifier in classifiers:
        stats.write(classifier.__name__)
        for vectorizer in vectorizers:
            for ngram_range in ngram_ranges:
                stats.write(',')
                try:
                    tester = ClassificationTester(classifier(), vectorizer(ngram_range=ngram_range))
                    time1 = time()
                    tester.train(train_set)
                    time2 = time()
                    accuracy = tester.accuracy(test_set)
                    time3 = time()
                    accuracy_percent = round(accuracy * 100)
                    train_time = round((time2 - time1) / len(train_set) * 1000)
                    test_time = round((time3 - time2) / len(test_set) * 1000)
                    stats.write(f'{accuracy_percent}% ({train_time} ms / {test_time} ms)')
                except Exception as exception:
                    stats.write('-')
                    exceptions.write(f'{tester!r} -> {exception!r}\n')
                    exceptions.flush()
                stats.flush()
        stats.write('\n')
