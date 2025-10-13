import numpy as np
from matplotlib import pyplot 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC

input_file = 'income_data.txt'
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

models = [] 
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr'))) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('NB', GaussianNB())) 
models.append(('SVM', SVC(gamma='auto')))

results = [] 
names = [] 
for name, model in models: 
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True) 
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy') 
    results.append(cv_results) 
    names.append(name) 
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# Порівняння алгоритмів 
pyplot.boxplot(results, labels=names) 
pyplot.title('Algorithm Comparison') 
pyplot.show()

def evaluate_classification_metrics(name, model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"Model name: {name}")
    print(f"Accuracy: {round(accuracy * 100, 2)}%")
    print(f"Precision: {round(precision * 100, 2)}%")
    print(f"Recall: {round(recall * 100, 2)}%")
    print(f"F1 score: {round(f1 * 100, 2)}% \n")


model_LR = LogisticRegression(solver='liblinear', multi_class='ovr')
model_LR.fit(X_train, y_train)
evaluate_classification_metrics('LR', model_LR, X_test, y_test)

model_LDA = LinearDiscriminantAnalysis()
model_LDA.fit(X_train, y_train)
evaluate_classification_metrics('LDA', model_LDA, X_test, y_test)

model_KNN = KNeighborsClassifier()
model_KNN.fit(X_train, y_train)
evaluate_classification_metrics('KNN', model_KNN, X_test, y_test)

model_CART = DecisionTreeClassifier()
model_CART.fit(X_train, y_train)
evaluate_classification_metrics('CART', model_CART, X_test, y_test)

model_NB = GaussianNB()
model_NB.fit(X_train, y_train)
evaluate_classification_metrics('NB', model_NB, X_test, y_test)

model_SVM = SVC(gamma='auto')
model_SVM.fit(X_train, y_train)
evaluate_classification_metrics('SVM', model_SVM, X_test, y_test)
