import random
from math import sqrt

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Carregando os dados
diamonds_data = pd.read_csv('./datasets/diamonds.csv')

# Dividindo os dados em recursos (x) e variável alvo (y)
x = diamonds_data.drop('price', axis=1)
y = diamonds_data['price']

# Convertendo variáveis categóricas em variáveis dummy
x = pd.get_dummies(x)

# Dividindo o conjunto de dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Padronizando os dados
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

try:
    # Carregando o modelo salvo
    network = joblib.load('diamond-relu.pkl')
    print('network loaded')
    # Variavel para treinar novamente o modelo
    reTrein = 0
    for i in range(reTrein):
        print(f'train {i + 1}')
        network.partial_fit(x_train_scaled, y_train)  # 2.40607104
except:
    # Configurando e treinandoi o modelo MLP
    network = MLPRegressor(
        verbose=True,
        hidden_layer_sizes=(100, 50),
        max_iter=1000,
        activation='relu',
        solver='adam',
        random_state=42,
        tol=0.00001
    )
    network.fit(x_train_scaled, y_train)
joblib.dump(network, 'diamond-relu.pkl')

printIndex = random.randint(0, y_test.size)

# Fazendo previsões
print(f'index: {printIndex}')
print(
    f'Previsão: '
    f'{network.predict(x_test_scaled)[printIndex]}'
)  

print(f'Valor correto: {y_test.iloc[printIndex]}')

# Cálculo da performance
y_pred = network.predict(x_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE: ", mse)
print("MAE: ", mae)
print("RMSE: ", sqrt(mse))
print("Média de valores: ", y_test.mean())

# iterações: 1493

# Resultados:
# MSE:  3323.494797002967
# MAE:  25.268863099271645
# RMSE:  57.64975973066121
# Média de valores:  3906.0357804968485

# Exemplo de previsão
# index: 2262
# Previsão: 1847.2515651172114
# Valor correto: 1805
