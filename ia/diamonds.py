import random

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Importando a base
df = pd.read_csv('./datasets/diamonds.csv', sep=',')

# Tratando a tabela
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

# Separando camada de entrada e saida
entries = dfEntries.values
outings = df['price'].values

# Dividindo os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(entries, outings, test_size=0.2, random_state=42)

loaded = False

# Carregando o modelo salvo ou criando um novo
try:
    network = joblib.load('diamond.pkl')
    print('network loaded')
    reTrein = 10
    for i in range(reTrein):
        print(f'train {i + 1}')
        network.partial_fit(X_train, y_train)  # 2.40607104
    loaded = True
except:
    network = MLPClassifier(
        verbose=True,
        max_iter=100,
        tol=0.00001,
        activation='logistic',
        learning_rate_init=0.001,
    )
    network.fit(X_train, y_train)

# salvando modelo
joblib.dump(network, 'diamond.pkl')


def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements


if loaded:
    printIndex = random.randint(0, y_test.size)

    print(f'index: {printIndex}')
    print(
        f'Previsão: '
        f'{network.predict(X_test)[printIndex]}'
    ) 

    print(f'Valor correto: {y_test[printIndex]}')

    y_pred = network.predict(X_test)
    cm = confusion_matrix(y_pred, y_test)

    # teste de acurácia
    print('Acurácia: ', accuracy(cm)) 
