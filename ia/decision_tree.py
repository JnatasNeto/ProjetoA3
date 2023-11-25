import random

import pandas as pd
from sklearn import set_config
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('./datasets/diamonds.csv', sep=',')

dfEntries = df.drop('price', axis=1)
dfEntries = dfEntries.drop('Unnamed: 0', axis=1)

mapCut = {
    'Fair': 1,
    'Good': 2,
    'Very Good': 3,
    'Premium': 4,
    'Ideal': 5,
}
dfEntries['cut'] = df['cut'].map(mapCut)

mapColor = {
    'D': 1,
    'E': 2,
    'F': 3,
    'G': 4,
    'H': 5,
    'I': 6,
    'J': 7,
}
dfEntries['color'] = df['color'].map(mapColor)

mapClarity = {
    'I1': 1,
    'SI2': 2,
    'SI1': 3,
    'VS2': 4,
    'VS1': 5,
    'VVS2': 6,
    'VVS1': 7,
    'IF': 8,
}
dfEntries['clarity'] = df['clarity'].map(mapClarity)

entries = dfEntries.values
outings = df['price'].values

# Divida os dados em treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(entries, outings, test_size=0.2, random_state=42)

set_config(print_changed_only=False)

dtr = DecisionTreeRegressor()
print(dtr)

DecisionTreeRegressor(
    ccp_alpha=0.0, criterion='mse', max_depth=None,
    max_features=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=2, min_samples_split=4,
    min_weight_fraction_leaf=0.0,
    random_state=None, splitter='best'
)

dtr.fit(x_train, y_train)

score = dtr.score(x_train, y_train)
print("R-squared:", score)

ypred = dtr.predict(x_test)

mse = mean_squared_error(y_test, ypred)
mae = mean_absolute_error(y_test, ypred)
print("MSE: ", mse)
print("MAE: ", mae)
print("RMSE: ", mse ** (1 / 2.0))
print("Média de valores: ", y_test.mean())

# x_ax = range(len(y_test))
# plt.plot(x_ax, y_test, linewidth=1, label="original")
# plt.plot(x_ax, ypred, linewidth=1.1, label="predicted")
# plt.title("y-test and y-predicted data")
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend(loc='best', fancybox=True, shadow=True)
# plt.grid(True)
# plt.show()

printIndex = random.randint(0, y_test.size)

print(f'index: {printIndex}')
print(
    f'Previsão: '
    f'{dtr.predict(x_test)[printIndex]}'
)  # 1056  1093

print(f'Valor correto: {y_test[printIndex]}')

# Resultados:
# MSE:  536837.5160131628
# MAE:  351.1618001483129
# RMSE:  732.6919652986259
# Média de valores:  3906.0357804968485

# Exemplo de previsão
# index: 249
# Previsão: 11899.0
# Valor correto: 11792
