import numpy as np
from pandas import read_csv 
from pandas.plotting import scatter_matrix 
from matplotlib import pyplot 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv" 
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 
'class'] 
dataset = read_csv(url, names=names) 

print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('class').size())

# Діаграма розмаху 
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False) 
pyplot.show() 

# Гістограма розподілу атрибутів датасета 
dataset.hist() 
pyplot.show() 

#Матриця діаграм розсіювання 
scatter_matrix(dataset) 
pyplot.show() 

# Розділення датасету на навчальну та контрольну вибірки 
array = dataset.values 
# Вибір перших 4-х стовпців 
X = array[:,0:4] 
# Вибір 5-го стовпця 
Y = array[:,4] 
# Розділення X і Y на навчальний та контрольний набори
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1) 

# Завантажуємо алгоритми моделі 
models = [] 
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr'))) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('NB', GaussianNB())) 
models.append(('SVM', SVC(gamma='auto'))) 

# оцінюємо модель на кожній ітерації 
results = [] 
names = [] 
for name, model in models: 
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True) 
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy') 
    results.append(cv_results) 
    names.append(name) 
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Порівняння алгоритмів 
pyplot.boxplot(results, labels=names) 
pyplot.title('Algorithm Comparison') 
pyplot.show()

# Створюємо прогноз на контрольній вибірці 
model = SVC(gamma='auto') 
model.fit(X_train, Y_train) 
predictions = model.predict(X_validation)

# Оцінюємо прогноз 
print("Точність (SVC):", accuracy_score(Y_validation, predictions)) 
print("Матриця неточностей (SVC): \n", confusion_matrix(Y_validation, predictions)) 
print("Звіт про класифікацію (SVC):\n", classification_report(Y_validation, predictions))

# Нова квітка
X_new = np.array([[5, 2.9, 1, 0.2]])
print("Форма масиву X_new: {}".format(X_new.shape))

# Прогноз для нової квітки
prediction1 = model.predict(X_new)
print("Прогноз (SVC): {}".format(prediction1))

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions_lda = model.predict(X_validation)

print("Точність  (LDA):", accuracy_score(Y_validation, predictions_lda)) 
prediction2 = lda.predict(X_new)  
print("Прогноз (LDA): {}".format(prediction2))  
print("Спрогнозована мітка:", prediction2[0])
