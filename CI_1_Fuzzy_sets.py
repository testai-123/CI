'''
Implement Union, Intersection, Complement and Difference operations on fuzzy sets. Also
create fuzzy relations by Cartesian product of any two fuzzy sets and perform max-min
composition on any two fuzzy relations
'''

# Union

A = {"a": 0.2, "b": 0.3, "c": 0.6, "d": 0.6}
B = {"a": 0.9, "b": 0.9, "c": 0.4, "d": 0.5}

Y = {}

for key in A:
    Y[key] = max(A[key], B[key])

print('Fuzzy Set Union is:', Y)


# Intersection
A = {"a": 0.2, "b": 0.3, "c": 0.6, "d": 0.6}
B = {"a": 0.9, "b": 0.9, "c": 0.4, "d": 0.5}

Y = {}

for key in A:
    Y[key] = min(A[key], B[key])

print('Fuzzy Set Intersection is:', Y)


#COMPLEMENT
for A_key in A:
    Y[A_key]= 1-A[A_key]
         
print('Fuzzy Set Complement is :', Y)


# Difference
A = {"a": 0.2, "b": 0.3, "c": 0.6, "d": 0.6}
B = {"a": 0.9, "b": 0.9, "c": 0.4, "d": 0.5}

Y = {}

for key in A:
    B_complement = 1 - B[key]  
    Y[key] = min(A[key], B_complement)  

print('Fuzzy Set Difference is:', Y)


# Cartesian
A = {"a": 0.2, "b": 0.3, "c": 0.6, "d": 0.6}
B = {"x": 0.9, "y": 0.9, "z": 0.4}

cartesian_product = {}

for key_A in A:
    for key_B in B:
        cartesian_product[(key_A, key_B)] = min(A[key_A], B[key_B])

print('Fuzzy Set Cartesian Product is:', cartesian_product)


#MAX MIN COMPOSITION

import numpy as np

def max_min_composition(R1, R2):
    result = np.zeros((len(R1), len(R2[0])))
    for i in range(len(R1)):
        for j in range(len(R2[0])):
            max_min = 0
            for k in range(len(R1[0])):
                max_min = max(min(R1[i][k], R2[k][j]), max_min)
            result[i][j] = max_min
    
    result = np.clip(result, 0, 1)
    return result


R1 = np.array([[0.3, 0.4, 0.7], [0.3, 0.5, 0], [0.4, 0.4, 1]])
R2 = np.array([[0.3, 0.3, 0], [0.3, 0.5, 0.1], [0.4, 0.4, 0]])
result = max_min_composition(R1, R2)
print("Max-Min Composition Result:")
print(result)