import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Wygeneruj syntetycznyproblem zadania klasykacji, spełniający następujące
# wymagania: -problem składa się z dwóch atrybutów, - oba atrybutyproblemu sąinformatywne, a więc niema cech
# redundantnych ani zbędnych, - problem jest dychotomią, - szum etykiet (błędnych przypisań klasy) stanowi 5% ogółu
# wzorców, - próbka problemu składa się z trzystu wzorców równomiernie rozłożonych po jego klasach.
X, y = ds.make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.05)

# Podziel wygenerowany w pierwszym zadaniu zbiór danych na część testową i uczącą, przyjmując 30% do testo-
# wania i 70% do uczenia.
X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.3, train_size=0.7)

# Zainicjalizuj gaussowski, naiwny klasykator Bayesa ze standardowymi hiperparametrami i wyucz (dopasuj) go
# na podstawie zbioru uczącego.
clf = GaussianNB()
clf.fit(X_train, y_train)

# Wyznacz macierz wsparć dla zbioru testowego wyuczonego klasyfikatora.
prob = clf.predict_proba(X_test)

# Na podstawie macierzy wsparć wyznacz predykcję klasyfikatora dla zbioru testowego.
y_pred = np.argmax(prob, axis=1)
# alternatywnie
# y_pred = clf.predict(X_test)

# Na podstawie wyznaczonej predykcji wylicz wartość metryki accuracy klasyfikatora.
accu = accuracy_score(y_test, y_pred)

# Na podzielonej na dwie części ilustracji, w formie scatterplota, przedstaw wsparcia klasyfikatora wyznaczone na
# zbiorze testowym (wzorce w dziedzinie wsparć). Dla lewej ilustracji przyjmij kolory dla etykiet rzeczywistych,
# dla prawej – etykiet będących wynikiem predykcji.
plt.subplot(121)
plt.scatter(prob[:,0],np.arange(1,prob[:,0].shape[0]+1,1) , c=y_test)
plt.scatter(prob[:,1],np.arange(1,prob[:,1].shape[0]+1,1) , c=y_test)

plt.subplot(122)
plt.scatter(prob[:,0],np.arange(1,prob[:,0].shape[0]+1,1), c=y_pred)
plt.scatter(prob[:,1],np.arange(1,prob[:,1].shape[0]+1,1), c=y_pred)

plt.savefig("plot2.png")

# print(prob[:,0].shape)
# print(np.arange(1,prob[:,0].shape[0]+1,1).shape)
