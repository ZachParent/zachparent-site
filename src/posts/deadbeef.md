---
layout: post
title: Deadbeef Decrementation
excerpt: How to decrement from a magic debug value to zero.
date: 2016-10-09
updatedDate: 2016-10-09
tags:
  - post
  - ai
published: true
---

```python
ONE_WORD_DESCRIPTION = "Fibonacci sequence generator."
# Fibonacci sequence generator.
def fibonacci(n):
  """Fibonacci sequence generator."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

```
