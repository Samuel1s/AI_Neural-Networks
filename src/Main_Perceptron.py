# Neural Network, Perceptron analysis.

import csv
import math
import random

import Array_Operations as arrop

INDEX_OF_PCT = lambda x, pct: int(round((len(x) * pct)))

DATA_TYPES_MAP = { 'Iris-setosa': [0, 0, 1], 'Iris-versicolor': [0, 1, 0], 'Iris-virginica': [1, 0, 0] }

def load_and_arrange_data():
    samples, labels, temp = [], [], []

    with open('iris.data', newline='') as csvfile:
        for row in csv.reader(csvfile, delimiter=','):
            temp.append(row)

    # random.shuffle(temp) # Shuffle data in random order.

    for row in temp:
        s, l = row[:(len(row) - 1)], row[(len(row) - 1):][0]

        samples.append([float(i) for i in s])
        labels.append(DATA_TYPES_MAP[l])

    return samples, labels

def step_func(x):
    result = []

    for element in x:
        result.append(1 if element >= 0 else 0)

    return result

def sigmoid_func(x):
    result, temp = [], []

    h_index = 0
    h_element = math.exp(x[h_index]) / (1 + math.exp(x[h_index]))

    for i, element in enumerate(x):
        new_element = math.exp(element) / (1 + math.exp(element))

        if new_element > h_element:
            h_index = i
            h_element = new_element

        temp.append(new_element)

    for i, element in enumerate(temp):
        result.append(1 if i == h_index else 0)

    return result

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

            y = activation_func(arrop.add_to_const(bias, arrop.multiply(weights, x)))
            error = arrop.sub(d, y)
            weights = arrop.add_to_const(weights, alpha * arrop.multiply(error, x))
            bias = arrop.add(bias, arrop.multiply_by_const(error, alpha))

            E = E + arrop.sum_squared(error)

        it = it + 1

    for i in range(len(s_test)):
        x, d = s_test[i], l_test[i]

        y = activation_func(arrop.add_to_const(bias, arrop.multiply(weights, x)))
        error = arrop.sub(d, y)

        hits += 1 if error.count(1) == 0 else 0 # Check if error only contains zeros.

    acuracy = (hits / len(s_test)) * 100

    return acuracy

def main():
    samples, labels = load_and_arrange_data()

    print('[STEP FUNCTION] : ACURACY =>', perceptron(0.1, samples, labels, step_func))
    print('[SIGMOID FUNCTION] : ACURACY =>', perceptron(0.1, samples, labels, sigmoid_func))

if __name__ == "__main__":
    main()