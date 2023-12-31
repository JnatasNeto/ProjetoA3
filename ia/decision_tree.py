import random

import pandas as pd
from sklearn import set_config
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Carregando os dados
df = pd.read_csv('./datasets/diamonds.csv', sep=',')

# Dividindo os dados em recursos (dfEntries) e variável alvo (dfEntries)
dfEntries = df.drop('price', axis=1)
dfEntries = dfEntries.drop('Unnamed: 0', axis=1)

# Convertendo variáveis String para núméricas
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

# Dividindo o conjunto de dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(entries, outings, test_size=0.2, random_state=42)

set_config(print_changed_only=False)

dtr = DecisionTreeRegressor()
print(dtr)

# Criando modelo de árvore de decisão
DecisionTreeRegressor(
    ccp_alpha=0.0, criterion='mse', max_depth=20,
    max_features=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=1, min_samples_split=2,
    min_weight_fraction_leaf=0.0,
    random_state=None, splitter='best'
)

# Treinando modelo
dtr.fit(x_train, y_train)

score = dtr.score(x_train, y_train)
print("R-squared:", score)

y_pred = dtr.predict(x_test)

# Printando desempenho do modelo
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MSE: ", mse)
print("MAE: ", mae)
print("RMSE: ", mse ** (1 / 2.0))
print("MAPE: ", mape * 100, '%')
print("Média de valores: ", y_test.mean())

# Pegando exemplo de previsão
printIndex = random.randint(0, y_test.size)

print(f'index: {printIndex}')
print(
    f'Previsão: '
    f'{dtr.predict(x_test)[printIndex]}'
)  # 1056  1093

print(f'Valor correto: {y_test[printIndex]}')

# Resultados:
# MSE:  533111.3800055617
# MAE:  351.72080088987764
# RMSE:  730.1447664713908
# MAPE:  8.315084855388413 %
# Média de valores:  3906.0357804968485

# Exemplo de previsão
# index: 8434
# Previsão: 979.0
# Valor correto: 1046
