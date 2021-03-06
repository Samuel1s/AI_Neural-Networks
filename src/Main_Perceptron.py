#*******************************************************************************
#*                      Redes Neurais-Algoritmo Perceptron                     *
#*-----------------------------------------------------------------------------*
#* @AUTHOR: Samuel Filipe dos Santos e Giulio Castro                           *
#* @TEACHER: Rogério Martins Gomes.                                            *
#* @LANGUAGE: Python                                                           *
#* @DISCIPLINE: Inteligência Artificial                                        *
#* @CODING: UTF-8                                                              *
#* @DATE: 07 de março de 2021                                                  *
#*******************************************************************************


import csv
import math
import random

INDEX_OF_PCT = lambda x, pct: int(round((len(x) * pct)))

# Transformação das saídas.
DATA_TYPES_MAP = { 'Iris-setosa': [0, 0, 1], 'Iris-versicolor': [0, 1, 0], 'Iris-virginica': [1, 0, 0] }

# Leitura dos dados.
def load_and_arrange_data():
    samples, labels, temp = [], [], []

    with open('iris.data', newline='') as csvfile:
        for row in csv.reader(csvfile, delimiter=','):
            temp.append(row)

    for row in temp:
        s, l = row[:(len(row) - 1)], row[(len(row) - 1):][0]

        samples.append([float(i) for i in s])
        labels.append(DATA_TYPES_MAP[l]) # Transformação atribuída.

    return samples, labels

# Operações para Arrays.
def multiply(arr1, arr2):
    result = 0

    for i in range(len(arr1)):
        result += arr1[i] * arr2[i]

    return result

def multiply_by_const(arr, const):
    result = []

    for element in arr:
        result.append(element * const)

    return result

def add(arr1, arr2):
    result = []

    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    return result

def add_to_const(arr, const):
    result = []

    for element in arr:
        result.append(element + const)

    return result

def sub(arr1, arr2):
    result = []

    for i in range(len(arr1)):
        result.append(arr1[i] - arr2[i])

    return result

def sum_squared(arr):
    result = 0

    for element in arr:
        result = element ** 2

    return result

# Função Degrau.
def step_function(x):
    result = []

    for element in x:
        result.append(1 if element >= 0 else 0)

    return result

# Função Sigmoidal.
def sigmoid_function(x):
    result, temp = [], []

    h_index = 0
    h_element = 1 / (1 + math.exp(x[h_index]))

    for i, element in enumerate(x):
        new_element = 1 / (1 + math.exp(element))

        if new_element > h_element:
            h_index = i
            h_element = new_element

        temp.append(new_element)

    for i, element in enumerate(temp):
        result.append(1 if i == h_index else 0)

    return result

# Algoritmo Perceptron.
def perceptron(alpha, samples, labels, activation_func, max_it=100):
    weights, bias = [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]

    E, it, hits = 1, 1, 0

    s_train, l_train = samples[:INDEX_OF_PCT(samples, 0.7)], labels[:INDEX_OF_PCT(labels, 0.7)]
    s_test, l_test = samples[INDEX_OF_PCT(samples, 0.3):], labels[INDEX_OF_PCT(labels, 0.3):]

    while it < max_it and E > 0:
        error = []
        E = 0

        for i in range(len(s_train)):
            x, d = s_train[i], l_train[i]

            y = activation_func(add_to_const(bias, multiply(weights, x)))
            error = sub(d, y)
            weights = add_to_const(weights, alpha * multiply(error, x))
            bias = add(bias, multiply_by_const(error, alpha))

            E = E + sum_squared(error)

        it = it + 1

    for i in range(len(s_test)):
        x, d = s_test[i], l_test[i]

        y = activation_func(add_to_const(bias, multiply(weights, x)))
        error = sub(d, y)

        hits += 1 if error.count(1) == 0 else 0 # Check if error only contains zeros.

    acuracy = (hits / len(s_test)) * 100

    return acuracy

def main():
    samples, labels = load_and_arrange_data()
    
    print('-------------------------------------------------------')
    print('TAXA DE ACERTO PARA AMBAS AS FUNÇÕES:')
    print('TAXA DE ACERTO P/ FUNÇÃO DEGRAU:', perceptron(0.1, samples, labels, step_function))
    print('TAXA DE ACERTO P/ FUNÇÃO SIGMOIDAL:', perceptron(0.1, samples, labels, sigmoid_function))
    print('-------------------------------------------------------')

if __name__ == "__main__":
    main()