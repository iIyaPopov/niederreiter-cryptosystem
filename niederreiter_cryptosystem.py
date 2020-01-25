# Обратная матрица по модулю: https://planetcalc.ru/3324/

import numpy as np
import math
import collections
import random
import time
import fractions


def get_field_generator(gf_modulo):
    for i in range(2, gf_modulo):
        elements = [0] * gf_modulo
        for j in range(gf_modulo):
            # print(i, j, int(math.pow(i, j) % gf_modulo))
            elements[int(math.pow(i, j) % gf_modulo)] = 1
        if collections.Counter(elements)[0] == 1:
            return i


def get_h_matrix(n, k, field_generator, modulo):
    array = np.zeros((n-k, n), dtype=int)
    for i in range(n-k):
        for j in range(n):
            deg = (i + 1) * j
            array[i][j] = math.pow(field_generator, deg) % modulo
    return array


def get_rand_matrix(v_size, h_size, min_value, max_value):
    return np.random.randint(min_value, max_value, (v_size, h_size), dtype=int)


def get_permutation(size):
    vector = np.zeros(size, dtype=int)
    values = [0] * size
    i = 0
    while collections.Counter(values)[0] > 0:
        rand = random.randint(0, size-1)
        if values[rand] == 0:
            vector[i] = rand
            values[rand] = 1
            i += 1
    return vector


def get_inverse_permutation(permutation):
    res = [0] * len(permutation)
    for i in range(len(permutation)):
        res[permutation[i]] = i
    return res


def permute_matrix(matrix, permutation):
    t_matrix = np.transpose(matrix)
    new_matrix = []
    for i in permutation:
        new_matrix.append(t_matrix[i])
    return np.transpose(new_matrix)


def matrix_do_modulo(matrix, modulo):
    res = []
    for row in matrix:
        res.append(vector_do_modulo(row, modulo))
    return np.array(res)


def vector_do_modulo(vector, modulo):
    res = []
    for i in vector:
        res.append(i % modulo)
    return np.array(res)


def encrypt_message(message, bhp_matrix):
    return np.dot(message, np.transpose(bhp_matrix))


def decrypt_message(message, inv_p_matrix, h_matrix):
    perm_message = []
    for i in range(len(message)):
        perm_message.append(message[inv_p_matrix[i]])
    return np.dot(perm_message, np.transpose(h_matrix))


def get_w_matrix(t, syndrome):
    for i in range(t, 0, -1):
        res = []
        for j in range(i):
            res.append(syndrome[j:j+i])
        matrix = np.array(res)
        print('det W =\n', np.linalg.det(matrix) % modulo)
        if np.linalg.det(matrix) % modulo != 0:
            return matrix


def get_y_vector(from_pos, to_pos, syndrome, modulo):
    res = []
    for i in range(from_pos, to_pos):
        res.append(-syndrome[i] % modulo)
    return np.array(res)


def get_inv_w_matrix(w_matrix):
    print('W =')
    for i in range(len(w_matrix)):
        for j in range(len(w_matrix[0])):
            print(w_matrix[i][j] % modulo, end=' ')
        print()
    input('Ожидается обратная матрица W в файле\n')
    return np.genfromtxt("inverse_w_matrix.txt", delimiter='\t', dtype=int)


def get_l_vector(inverse_w_matrix, y_vector):
    return np.dot(inverse_w_matrix, y_vector)


def find_roots(l_vector):
    res = []
    poly = list(l_vector)
    poly.append(1)
    poly.reverse()
    print('poly indexes =\n', poly)
    for i in range(modulo):
        x = math.pow(generator, i) % modulo
        sum = 0
        for j in range(len(poly)):
            sum += poly[j] * math.pow(x, j)
        sum %= modulo
        if sum == 0:
            res.append(i)
    return res


def inverse_roots(roots):
    return [-i % (modulo - 1) for i in roots]


def get_inv_x_matrix(inv_roots):
    res = np.zeros((len(inv_roots), len(inv_roots)), dtype=int)
    print('X =')
    for i in range(len(inv_roots)):
        for j in range(len(inv_roots)):
            res[i][j] = math.pow(math.pow(generator, inv_roots[j]) % modulo, i+1) % modulo
            print(res[i][j], end=' ')
        print()
    input('Ожидается обратная матрица для X\n')
    return np.genfromtxt("inverse_x_matrix.txt", delimiter='\t', dtype=int)


def get_errors(inv_x_matrix, syndrom):
    return vector_do_modulo(np.dot(inv_x_matrix, syndrom[0:t]), modulo)


def get_error_vector(errors, inv_roots):
    res = [0] * n
    for i in range(len(inv_roots)):
        res[inv_roots[i]] = errors[i]
    return res


def get_message(size, error_count):
    res = [0] * size
    i = 0
    while i < error_count:
        ind = random.randint(0, size-1)
        if res[ind] == 0:
            res[ind] = random.randint(0, modulo - 1)
            i += 1
    return res


n = 36
k = 17
t = math.floor(((n - k + 1) - 1) / 2)
modulo = n + 1
generator = get_field_generator(modulo)
h_matrix = get_h_matrix(n, k, generator, modulo)
while True:
    b_matrix = get_rand_matrix(n-k, n-k, 0, modulo)
    if np.linalg.det(b_matrix) != 0:
        break
#b_matrix = [[3, 6, 6, 2, 1, 0],
 #           [6, 2, 4, 5, 3, 7],
  #          [1, 5, 10, 6, 7, 0],
   #         [9, 0, 0, 1, 5, 4],
    #        [6, 1, 1, 5, 8, 2],
     #       [10, 7, 0, 9, 3, 1]]
p_matrix = get_permutation(n)
bh_matrix = matrix_do_modulo(np.dot(b_matrix, h_matrix), modulo)
print('BH=\n', bh_matrix)
#permutation = [3, 4, 7, 5, 0, 9, 1, 6, 8, 2]
permutation = get_permutation(n)
bhp_matrix = permute_matrix(bh_matrix, permutation)
print('BHP =\n', bhp_matrix)
# ibh_matrix = permute_matrix(bhp_matrix, get_inverse_permutation(permutation))
# print('iBHP =\n', ibh_matrix)
#message = np.array([3, 0, 0, 5, 0, 0, 7, 0, 0, 0])
message = get_message(n, error_count=t)
print('message =\n', message)
encrypted_message = encrypt_message(message, bhp_matrix)
print('encrypted_message=\n', vector_do_modulo(encrypted_message, modulo))
decrypted_message = decrypt_message(message, get_inverse_permutation(permutation), h_matrix)
print('decrypted_message=\n', vector_do_modulo(decrypted_message, modulo))
w_matrix = get_w_matrix(t, decrypted_message)
print('W =\n', matrix_do_modulo(w_matrix, modulo))
y_vector = get_y_vector(t, 2*t, decrypted_message, modulo)
print('y=\n', y_vector)
inv_w_matrix = get_inv_w_matrix(w_matrix)
print('inv W =\n', inv_w_matrix)
l_vector = vector_do_modulo(get_l_vector(inv_w_matrix, y_vector), modulo)
print('L =\n', l_vector)
roots = find_roots(l_vector)
print('roots =\n', roots)
inv_roots = inverse_roots(roots)
print('inv roots =\n', inv_roots)
inv_x_matrix = get_inv_x_matrix(inv_roots)
print('inv x =\n', inv_x_matrix)
errors = get_errors(inv_x_matrix, decrypted_message)
print('errors =\n', errors)
error_vector = get_error_vector(errors, inv_roots)
print('error vector =\n', error_vector)
finally_message = permute_matrix(error_vector, permutation)
print('finally message =\n', np.array(finally_message))
print('message =\n', np.array(message))
print(finally_message == message)
