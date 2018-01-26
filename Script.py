
# coding: utf-8

# In[63]:

from time import time

import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.svm import LinearSVC


# Считываю данные из файла __train.csv__, используя средства библиотеки _pandas_, и визуализирую их. Таким образом имеется ___489517___ классифицированных объявлений:

# In[64]:

X_reader = []
y_reader = []
data_train = pd.read_csv("data/train.csv")
data_train


# Данные из файла __train.csv__ классифицированы по категориям. Сами категории имеют иерархичную структуру, но различные категории имеют различное количество уровней иерархии. Считываю данные из файла __category.csv__ и визуализирую их с помощью средств _pandas_ в виде объекта __DataFrame__. Так как категории имеют различную глубину иерархии, я продолжаю иерархию (там где это нужно) повторением последнего элемента, например: _Бытовая электроника|Ноутбуки|Ноутбуки|Ноутбуки_. То есть для каждой категории уравниваю количество уровней иерархии до 4. Я делаю это, чтобы в дальнейшем мне было удобно реализовать обучение классификатора на разных уровнях иерархии:

# In[65]:

data_categories_read = pd.read_csv("data/category.csv")

data_categories = data_categories_read.name.str.split('|', expand=True).reset_index()
data_categories.columns = ['category_id','hierarchy 1','hierarchy 2','hierarchy 3','hierarchy 4']
data_categories = data_categories.transpose().fillna(method='ffill').transpose()
hierarchy_1 = data_categories['hierarchy 1']
hierarchy_2 = hierarchy_1.str.cat(data_categories['hierarchy 2'], sep='|')
hierarchy_3 = hierarchy_2.str.cat(data_categories['hierarchy 2'], sep='|')
hierarchy_4 = hierarchy_3.str.cat(data_categories['hierarchy 3'], sep='|')
data_categories = pd.DataFrame({ 'category_id' : data_categories['category_id'],
                                 'hierarchy_1' : hierarchy_1,
                                 'hierarchy_2' : hierarchy_2,
                                 'hierarchy_3' : hierarchy_3,
                                 'hierarchy_4' : hierarchy_4})
data_categories


# Добавляю столбцы с уровнями категориальной иерархии к исходным данным, чтобы сопоставить каждому __'item_id'__ свое значение категории __'hierarchy_n'__ (n - уровень иерархии) по каждому иерархическому уровню:

# In[66]:

data_train = pd.merge(data_train, data_categories, on="category_id")
data_train = data_train.sort_values(by='item_id')
data_train


# Предобратка данных для дальнейшего их использования при обучении классификатора. В качестве данных для обучения я выбрал текстовые данные, которые помещаю в __X_train_read__. Для их предобработки я использую средства библиотеки __scikit_learn: sklearn.feature_extraction.text.CountVectorizer().fit_transform()__
# Данный метод преобразует текст в матрицу объектов и признаков (__[n_samples, n_features]__), с которой будет работать классификатор. В данном случае количество признаков равно размеру словаря, который рассматриваемый метод образует при анализе данных.
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# На выходе из CountVectorizer().fit_transform имеем __X_train__, который представляет из себя матрицу объектов и признаков. 
# В данном случае в качестве сырых текстовых данных выбраны названия (__'title'__) объявлений, ниже будет пояснено - почему.

# In[67]:

X_train_read = data_train['title']

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train_read)
X_train


# В качестве классификатора, для начала, мною был выбран _наивный байесовский классификатор_, как относительно простой и популярный метод классификации текста:
# https://en.wikipedia.org/wiki/Naive_Bayes_classifier
# В качестве реализации наивного байесовского классификатора в данной задаче была выбрана модель __Multinomial naive Bayes__, подходящая для классификации дискретных функций (в данном случае подсчет слов в тексте). Однако, ниже представлено, что также можно использовать классификатор с _моделью событий Бернулли_ (__Bernoulli naive Bayes__), и его результаты практически не отличаются от результатов __MultinomialNB__.
# В качестве классов для обучения классификатора здесь выбраны числовые значения категорий (__'category id'__).
# В качестве метрики оценки качества работы классификатора используется доля правильных ответов алгоритма - __accuracy__. Однако, для оценивания обобщающей способности обучаемого алгоритма, также реализовывалась _кросс-валидация методом контроля по K блокам_ (__K-fold CV__), где K=10 (обычно используемое значение). 
# https://en.wikipedia.org/wiki/Cross-validation_(statistics)
# Также, в качестве дополнительной информации, выводится длительность обучения классификатора и длительность кросс-проверки в секундах:

# In[68]:

y_reader = data_train['category_id'].tolist()
y_train = y_reader

clf = MultinomialNB(alpha=0.1)
t0 = time()
clf.fit(X_train, y_train)
duration = time() - t0

print('learning duration', duration)
print('accuracy with train data', clf.score(X_train, y_train))
t0 = time()
print('accuracy with cross validation (K Folds, n=10)', cross_val_score(clf, X_train, y_train, cv=10).mean())
duration = time() - t0
print('duration of cross validation', duration)


# Пример реализации наивного байесовского классификатора на основе _модели собыйти Бернулли_ (__Bernoulli naive Bayes__):

# In[69]:

y_reader = data_train['category_id'].tolist()
y_train = y_reader

clf_bernoulli = BernoulliNB(alpha=0.1)
t0 = time()
clf_bernoulli.fit(X_train, y_train)
duration = time() - t0

print('learning duration', duration)
print('accuracy with train data', clf_bernoulli.score(X_train, y_train))
t0 = time()
print('accuracy with cross validation (K Folds, n=10)', cross_val_score(clf_bernoulli, X_train, y_train, cv=10).mean())
duration = time() - t0
print('duration of cross validation', duration)


# Ниже представлена реализация обучения наивного байсовского классификатора __MultiNomialNB__ с использованием в качестве обучающих данных описания (__'description'__) объявлений. Однако метрика __accuracy__ в этом случае ниже (примерно на ___8%___), чем при обучении на __'title'__:

# In[70]:

X_train_read = data_train['description']

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train_read)

clf_with_description = MultinomialNB(alpha=0.1)
t0 = time()
clf_with_description.fit(X_train, y_train)
duration = time() - t0

print('learning duration', duration)
print('accuracy with train data', clf_with_description.score(X_train, y_train))
t0 = time()
print('accuracy with cross validation (K Folds, n=10)', cross_val_score(clf_with_description, X_train, y_train, cv=10).mean())
duration = time() - t0
print('duration of cross validation', duration)


# Размер матрицы __[n_samples, n_features]__ при обучении на __'description'__ значительно больше по количеству признаков (__features__), чем в случае обучения на __'title'__. Скорее всего это и является причиной ухудшения качества работы алгоритма в силу "наивности" байесовского классификатора (независимости условной вероятности для слов, использующихся в качестве признаков):

# In[71]:

X_train


# In[72]:

X_train_read = data_train['title']

X_train = vectorizer.fit_transform(X_train_read)
X_train


# Ниже реализован __scoring__ для неклассифицированных тестовых данных из файла __test.csv__. Результаты предсказаний классификатора записаны в файл __test_scoring.csv__:

# In[73]:

data_test = pd.read_csv("data/test.csv")
X_test_read = data_test['title']
X_test = vectorizer.transform(X_test_read)
item_id = data_test['item_id']
test_scoring = pd.DataFrame({ 'item_id' : item_id,
                              'category_id' : clf.predict(X_test)})
test_scoring = test_scoring.reindex(columns=['item_id','category_id'])
test_scoring.to_csv("data/test_scoring.csv", index=False)
test_scoring


# Визуализировано истинное распределение объявлений из __train.csv__ по всем категориям. Можно сделать вывод, что в обучающих данных нет уклона в сторону какого-то конкретного класса:

# In[74]:

data_train.hierarchy_4.value_counts(sort=False).plot(kind="barh", figsize=(10,20), title='Categories', fontsize=14)


# Сравнение результата работы обученного классификатора на тренировочных данных (непрозрачные столбцы - истинное распределение, полупрозрачные - результат работы классификатора):

# In[75]:

predicted_train = pd.Series(clf.predict(X_train), name='predicted')
data_train.category_id.value_counts(sort=False).plot(kind="barh", figsize=(10,20), width=.3, title='Categories', fontsize=14, legend=True)
predicted_train.value_counts(sort=False).plot(kind="barh", figsize=(10,20), width=.3, title='Categories', fontsize=14, alpha=.6, position=1.5, legend=True)


# Обучение классификатора на __3__ уровне иерархии (__'hierarchy_3'__), __accuracy__ и визуализированное представление результата работы алгоритма для данного уровня:

# In[76]:

y_reader = data_train['hierarchy_3']
y_train = y_reader

clf = MultinomialNB(alpha=0.1)
clf.fit(X_train, y_train)

print('accuracy on train data', clf.score(X_train, y_train))
print('accuracy with cross validation (K Folds, n=10)', cross_val_score(clf, X_train, y_train, cv=10).mean())

predicted_train = pd.Series(clf.predict(X_train), name='predicted')
data_train.hierarchy_3.value_counts(sort=False).plot(kind="barh", figsize=(10,20), width=.3, title='Categories', fontsize=14, legend=True)
predicted_train.value_counts(sort=False).plot(kind="barh", figsize=(10,20), width=.3, title='Categories', fontsize=14, alpha=.6, position=1.5, legend=True)


# Обучение классификатора на __2__ уровне иерархии (__'hierarchy_2'__), __accuracy__ и визуализированное представление результата работы алгоритма для данного уровня:

# In[77]:

y_reader = data_train['hierarchy_2']
y_train = y_reader

clf = MultinomialNB(alpha=0.1)
clf.fit(X_train, y_train)

print('accuracy on train data', clf.score(X_train, y_train))
print('accuracy with cross validation (K Folds, n=10)', cross_val_score(clf, X_train, y_train, cv=10).mean())

predicted_train = pd.Series(clf.predict(X_train), name='predicted')
data_train.hierarchy_2.value_counts(sort=False).plot(kind="barh", figsize=(10,20), width=.3, title='Categories', fontsize=14, legend=True)
predicted_train.value_counts(sort=False).plot(kind="barh", figsize=(10,20), width=.3, title='Categories', fontsize=14, alpha=.6, position=1.5, legend=True)


# Обучение классификатора на __1__ уровне иерархии (__'hierarchy_1'__), __accuracy__ и визуализированное представление результата работы алгоритма для данного уровня:

# In[78]:

y_reader = data_train['hierarchy_1']
y_train = y_reader

clf = MultinomialNB(alpha=0.1)
clf.fit(X_train, y_train)

print('accuracy on train data', clf.score(X_train, y_train))
print('accuracy with cross validation (K Folds, n=10)', cross_val_score(clf, X_train, y_train, cv=10).mean())

predicted_train = pd.Series(clf.predict(X_train), name='predicted')
data_train.hierarchy_1.value_counts(sort=False).plot(kind="barh", figsize=(10,20), width=.3, title='Categories', fontsize=14, legend=True)
predicted_train.value_counts(sort=False).plot(kind="barh", figsize=(10,20), width=.3, title='Categories', fontsize=14, alpha=.6, position=1.5, legend=True)


# Еще одним популярным алгоритмом для обучения при решении задачи классификации текста является _Метод Опорных Векторов_(__Support vector machine, SVM__ https://en.wikipedia.org/wiki/Support_vector_machine), суть которого заключается в построении оптимальной разделяющей гиперплоскости. Ниже данный алгоритм реализован с помощью __sklearn.svm.LinearSVC__. В качестве обучающих данных алгоритм все также принимает названия объявлений __'title'__, а в качестве классов - __'hierarchy_4'__. Дополнительно выводится время, затрачиваемое на обучение классификатора. Можно заметить, что оно значительно превышает аналогичное время для __MultinomialNB__ классификатора:

# In[79]:

y_reader = data_train['hierarchy_4']
y_train = y_reader

clf = LinearSVC()
t0 = time()
clf.fit(X_train, y_train);
duration = time() - t0
print('learning duration', duration)


# Вычисление __accuracy__ производится аналогично случаю с __MultinomialNB__. Также дополнительно выводится время, затраченное на кросс-проверку в секундах:

# In[27]:

print('accuracy on train data', clf.score(X_train, y_train))
t0 = time()
print('accuracy with cross validation (K Folds, n=10)', cross_val_score(clf, X_train, y_train, cv=10).mean())
duration = time() - t0
print('duration of cross_validation', duration)


# Обучение __LinearSVM__ классификатора на __3__ уровне иерархии (__'hierarchy_3'__), __accuracy__, результат кросс-проверки и время, затраченное на обучение и кросс-проверку в секундах:

# In[29]:

y_reader = data_train['hierarchy_3']
y_train = y_reader

clf = LinearSVC()
t0 = time()
clf.fit(X_train, y_train)
duration = time() - t0
print('learning duration', duration)


# In[30]:

print('accuracy on train data', clf.score(X_train, y_train))
t0 = time()
print('accuracy with cross validation (K Folds, n=10)', cross_val_score(clf, X_train, y_train, cv=10).mean())
duration = time() - t0
print('duration of cross_validation', duration)


# Обучение __LinearSVM__ классификатора на __2__ уровне иерархии (__'hierarchy_2'__), __accuracy__, результат кросс-проверки и время, затраченное на обучение и кросс-проверку в секундах:

# In[31]:

y_reader = data_train['hierarchy_2']
y_train = y_reader

clf = LinearSVC()
t0 = time()
clf.fit(X_train, y_train)
duration = time() - t0
print('learning duration', duration)


# In[32]:

print('accuracy on train data', clf.score(X_train, y_train))
t0 = time()
print('accuracy with cross validation (K Folds, n=10)', cross_val_score(clf, X_train, y_train, cv=10).mean())
duration = time() - t0
print('duration of cross_validation', duration)


# Обучение __LinearSVM__ классификатора на __1__ уровне иерархии (__'hierarchy_1'__), __accuracy__, результат кросс-проверки и время, затраченное на обучение и кросс-проверку в секундах:

# In[33]:

y_reader = data_train['hierarchy_1']
y_train = y_reader

clf = LinearSVC()
t0 = time()
clf.fit(X_train, y_train)
duration = time() - t0
print('learning duration', duration)


# In[34]:

print('accuracy on train data', clf.score(X_train, y_train))
t0 = time()
print('accuracy with cross validation (K Folds, n=10)', cross_val_score(clf, X_train, y_train, cv=10).mean())
duration = time() - t0
print('duration of cross_validation', duration)

