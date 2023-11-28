import random

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

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

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Criando o modelo k-NN para regressão e treinando-o
knn_regressor = KNeighborsRegressor(
    n_neighbors=5,
    metric='manhattan',
    weights='distance'
)
knn_regressor.fit(x_train_scaled, y_train)

y_pred = knn_regressor.predict(x_test_scaled)

# Printando desempenho do modelo
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MSE: ", mse)
print("MAE: ", mae)
print("RMSE: ", mse ** (1 / 2.0))
print("MAPE: ", mape * 100, '%')
print("Média de valores: ", y_test.mean())

printIndex = random.randint(0, y_test.size)

# Pegando exemplo de previsão
print(f'index: {printIndex}')
print(
    f'Previsão: '
    f'{knn_regressor.predict(x_test_scaled)[printIndex]}'
)  

print(f'Valor correto: {y_test[printIndex]}')

# Resultados:
# MSE:  414211.7366552523
# MAE:  327.6626639822504
# RMSE:  643.5928345275856
# MAPE:  9.327774021442123 %
# Média de valores:  3906.0357804968485

# Exemplo de previsão
# index: 9269
# Previsão: 529.1977361385503
# Valor correto: 432
