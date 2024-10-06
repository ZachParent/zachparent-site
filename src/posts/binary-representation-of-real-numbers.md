---
layout: post
title: Binary Representation of Real Numbers
excerpt: An introduction to binary representation of real numbers.
date: 2024-10-06
updatedDate: 2024-10-06
tags:
  - post
  - code
---


```python
import numpy as np
```


```python
a, b = -1, 2
epsilon = 10**-6
r = 0.637197
expected_encoded_r = '1000101110110101000110'

```


```python
k = int(np.ceil(np.log2((b - a)/(2*epsilon))))

print('k:\t\t', k)
print('expected_k:\t', len(expected_encoded_r))
```

    k:		 21
    expected_k:	 22



```python
def enc(r: float) -> str:
    # Original equation: r_encoded = floor((r - a) / (b - a) * (2^k - 1) + 1/2)
    normalized_r = (r - a) / (b - a)
    scaled_r = normalized_r * (2**k - 1)
    rounded_r = int(scaled_r + 1/2)
    binary_representation = f"{rounded_r:b}"
    return binary_representation

def dec(encoded_r: str) -> float:
    # Original equation: a + int(encoded_r, 2) * (b-a)/(2**k - 1)
    encoded_int = int(encoded_r, 2)
    scaling_factor = (b - a) / (2**k - 1)
    scaled_value = encoded_int * scaling_factor
    decoded_r = a + scaled_value
    return decoded_r
```


```python
print('r:\t\t\t', r)
print('expected_encoded_r:\t', expected_encoded_r)
print()
print('encoded_r:\t\t', enc(r))
print('dec(encoded_r):\t\t', dec(enc(r)))
print('rounded dec(encoded_r):\t', round(dec(enc(r)), -int(np.floor(np.log10(epsilon)))))
```

    r:			 0.637197
    expected_encoded_r:	 1000101110110101000110
    
    encoded_r:		 100010111011010100011
    dec(encoded_r):		 0.637196844671652
    rounded dec(encoded_r):	 0.637197

