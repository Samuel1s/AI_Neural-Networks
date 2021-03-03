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