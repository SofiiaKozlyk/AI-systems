import datetime
import json
import numpy as np
from sklearn import covariance, cluster
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# Вхідний файл із символічними позначеннями компаній 
input_file = 'company_symbol_mapping.json'

# Завантаження прив'язок символів компаній до їх повних назв
with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

# Завантаження архівних даних котирувань
start_date = datetime.datetime(2003, 7, 3)
end_date = datetime.datetime(2007, 5, 4)
quotes = yf.download(list(symbols), start=start_date, end=end_date)

# Вилучення котирувань, що відповідають відкриттю та 
# закриттю біржі, та обчислення різниці між ними
opening_quotes = quotes['Open']
closing_quotes = quotes['Close']
quotes_diff = opening_quotes - closing_quotes

# Видалення компаній з некоректними даними
quotes_diff.dropna(axis='columns', how='all', inplace=True)
quotes_diff.dropna(axis='rows', how='any', inplace=True)

# Нормалізація даних
X = quotes_diff.copy()
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Створення моделі графа та її навчання
edge_model = covariance.GraphicalLassoCV(assume_centered=True)
edge_model.fit(X)

# Створення моделі кластеризації на основі поширення подібності 
affinity_model = cluster.AffinityPropagation(preference=np.median(edge_model.covariance_), random_state=42)
affinity_model.fit(edge_model.covariance_)

labels = affinity_model.labels_
num_labels = len(labels)
valid_symbols = []
for symbol in quotes_diff.columns.tolist():
    valid_symbols.append(company_symbols_map[symbol])

valid_symbols = np.array(valid_symbols)

# Виведення результатів
print(f"Кількість компаній: {len(valid_symbols)}")
print(f"Кількість знайдених кластерів: {num_labels}")
print("\nРезультати кластеризації:")
for i in range(num_labels):
    cluster_companies = valid_symbols[labels == i]
    print(f"Кластер {i+1} ==> {', '.join(cluster_companies)}")
