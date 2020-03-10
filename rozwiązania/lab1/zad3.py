from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.05)

# Przygotuj obiekt stratyfikowanej walidacji krzyżowej z pięcioma foldami.
skf = StratifiedKFold(n_splits=5)

# Przygotuj zmienną, w której będziesz przechowywać wyniki eksperymentu. Pięciofoldowa walidacja krzyżowa
# generuje dla każdego algorytmu pięć wyników
result = np.arange(5.)

# W każdej pętli walidacji krzyżowej:
# • zainicjalizuj klasyfikator bazowy (gaussowski naiwny klasykator Bayesa),
# • zbuduj model klasyfikatora (wykorzysując zbiór uczący),
# • wyznacz predykcję (wykorzystując zbiór testowy),
# • oszacuj jakość modelu (metryką accuracy),
# • zapisz wynik w odpowiednim polu wektora przygotowanego w punkcie 2.
i = 0
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accu = accuracy_score(y_test, y_pred)
    result[i] = accu
    i += 1

print(result)
