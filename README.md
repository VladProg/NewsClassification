## I. ПОСТАНОВКА ЗАДАЧІ

Задача &mdash; класифікація новин по 6 категоріям:
* `polytics` (політика),
* `economy` (економіка),
* `society` (суспільство),
* `culture` (культура),
* `sports` (спорт),
* `technology` (технології).

Кожна новина &mdash; це текстовий файл.

## II. РЕПОЗИТОРІЙ

Всі коди програм, а також тестові і вихідні дані доступні в GitHub-репозиторії https://github.com/VladProg/NewsClassification.

## III. ЗАВАНТАЖЕННЯ ДАНИХ ДЛЯ ТЕСТУВАННЯ

Новини взяті з сайту https://www.ukrinform.ua, кожна категорія з відповідної сторінки:
* https://www.ukrinform.ua/rubric-polytics/block-lastnews,
* https://www.ukrinform.ua/rubric-economy/block-lastnews,
* https://www.ukrinform.ua/rubric-society/block-lastnews,
* https://www.ukrinform.ua/rubric-culture/block-lastnews,
* https://www.ukrinform.ua/rubric-sports/block-lastnews,
* https://www.ukrinform.ua/rubric-technology/block-lastnews.

Я завантажив новини за допомогою скрипта [scraper.py](scraper.py). 
Він записав по 1000 новин кожної категорії (всього 6000 статей) в директорію [articles](articles), а їх категорії &mdash; у файл [list.txt](list.txt).

## IV. РЕАЛІЗАЦІЯ РОЗВ’ЯЗКУ

Для класифікації я використав модуль [sklearn](https://scikit-learn.org/stable/modules/classes.html). 
В ньому є класи для векторизації текстів (тобто перетворення їх на вектори чисел) і для власне класифікації отриманих векторів.

Для класифікації я використовував класи:
* [sklearn.calibration.CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV),
* [sklearn.discriminant_analysis.LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis),
* [sklearn.dummy.DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier),
* [sklearn.ensemble.AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier),
* [sklearn.ensemble.BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier),
* [sklearn.ensemble.ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier),
* [sklearn.ensemble.GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier),
* [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier),
* [sklearn.gaussian_process.GaussianProcessClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier),
* [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression),
* [sklearn.linear_model.LogisticRegressionCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV),
* [sklearn.linear_model.PassiveAggressiveClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier),
* [sklearn.linear_model.Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron),
* [sklearn.linear_model.RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier),
* [sklearn.linear_model.RidgeClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV),
* [sklearn.linear_model.SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier),
* [sklearn.naive_bayes.BernoulliNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB),
* [sklearn.naive_bayes.CategoricalNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB),
* [sklearn.naive_bayes.ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB),
* [sklearn.naive_bayes.GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB),
* [sklearn.naive_bayes.MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB),
* [sklearn.neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier),
* [sklearn.neighbors.NearestCentroid](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid),
* [sklearn.neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier),
* [sklearn.svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC),
* [sklearn.svm.NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC),
* [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC),
* [sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier),
* [sklearn.tree.ExtraTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier).

Для векторизації я використовував класи:
* [sklearn.feature_extraction.text.CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer),
* [sklearn.feature_extraction.text.HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer),
* [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer).

З наявних 6000 статей я випадковим чином обираю деяку кількість статей для тренування класифікатора, а всі інші &mdash; для тестування. 
Обробка статей відбувається у класі [ClassificationTester](classification_tester.py). В ньому є методи:
* `train` для тренування векторизатора і класифікатора,
* `classify_many` для класифікації новин;
* `accuracy` для обчислення точності класифікації (це кількість статей, для яких класифікатор "вгадав" категорію, поділена на загальну кількість статей).

## V. ОТРИМАНІ РЕЗУЛЬТАТИ

Точність залежить від вибору векторизатора, класифікатора і кількості новин для тренування. 
Отримана статистика показана у таблицях [stats_10.csv](stats_10.csv), [stats_50.csv](stats_50.csv), [stats_500.csv](stats_500.csv), [stats_5000.csv](stats_5000.csv). 
Число в назві файлу означає кількість статей для тренування. 
Кожен рядок таблиці відповідає деякому класифікатору, а кожен стовпчик &mdash; деякому векторизатору. 
В комірках таблиці наведений результат класифікації: точність, середній час тренування, витрачений на одну статтю, і середній час класифікації, витрачений на одну статтю. 
Порожня комірка означає, що в якійсь функції було кинуто exception.

## VI. ВИСНОВКИ

Перший висновок, який можна зробити &mdash; чим більше даних для тренування, тим більша точність (ну це, мабуть, було і так очевидно):
* 10 статей &mdash; найкращий результат 36%,
* 50 статей &mdash; найкращий результат 58%,
* 500 статей &mdash; найкращий результат 87%,
* 5000 статей &mdash; найкращий результат 91%.

Порівняємо векторизатори. В середньому найкраще працює `TfidfVectorizer`, а найгірше &mdash; `HashingVectorizer`. Проте різниця незначна.

Порівняємо класифікатори. В середньому найкраще працюють:
* `CalibratedClassifierCV` (але він не працює для малої кількості тренувальних даних),
* `LogisticRegressionCV` (але він працює дуже довго),
* `PassiveAggressiveClassifier`,
* `Perceptron`,
* `RidgeClassifier`,
* `RidgeClassifierCV`,
* `SGDClassifier`,
* `ComplementNB` (але він не працює разом з `HashingVectorizer`),
* `GaussianNB` (але він не працює разом з `HashingVectorizer`),
* `MLPClassifier` (але він працює дуже довго),
* `LinearSVC`.

## VII. СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ

1. Укрінформ. URL: https://www.ukrinform.ua/.
2. scikit-learn. URL: https://scikit-learn.org/.
