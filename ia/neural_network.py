import random
from math import sqrt

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
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
    # Configurando e treinando o modelo MLP
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
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MSE: ", mse)
print("MAE: ", mae)
print("RMSE: ", sqrt(mse))
print("MAPE: ", mape * 100, '%')
print("Média de valores: ", y_test.mean())

# com 1493 iterações

# Resultados:
# MSE:  3323.494797002967
# MAE:  25.268863099271645
# RMSE:  57.64975973066121
# MAPE:  1.8696014488082793 %
# Média de valores:  3906.0357804968485


# Exemplo de previsão
# index: 10005
# Previsão: 2925.872249280787
# Valor correto: 2913


# com 10493 iterações (diamond-relu-10k.pkl)

# Resultados:
# MSE:  2576.665712026238
# MAE:  17.603135639596992
# RMSE:  50.76086792034035
# MAPE: 1,187004661096045 %
# Média de valores:  3906.0357804968485

# Exemplo de previsão
# index: 7793
# Previsão: 756.1300119369904
# Valor correto: 764