import time
from decimal import Decimal
import sys
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate

matplotlib.use('TkAgg')

sys.setrecursionlimit(999999)


# recursive method
# -------------------------------------------------------------------------------
# recursive calculation of the n-th Fibonacci number.
def recursive(n):
    if n <= 1:
        return n
    return recursive(n - 1) + recursive(n - 2)
# -------------------------------------------------------------------------------


# dynamic programming method
# -------------------------------------------------------------------------------
# uses a bottom-up approach to compute the n-th Fibonacci number.
def dynamic_programming(n):
    l = [0, 1]
    for i in range(2, n + 1):
        l.append(l[i - 1] + l[i - 2])
    return l[n]
# -------------------------------------------------------------------------------


# matrix power method
# -------------------------------------------------------------------------------
# multiplies two 2x2 matrices.
def multiply(mat1, mat2):
    x = mat1[0][0] * mat2[0][0] + mat1[0][1] * mat2[1][0]
    y = mat1[0][0] * mat2[0][1] + mat1[0][1] * mat2[1][1]
    z = mat1[1][0] * mat2[0][0] + mat1[1][1] * mat2[1][0]
    w = mat1[1][0] * mat2[0][1] + mat1[1][1] * mat2[1][1]

    mat1[0][0], mat1[0][1] = x, y
    mat1[1][0], mat1[1][1] = z, w

# raises a matrix to the n-th power.
def power(mat1, n):
    if n == 0 or n == 1:
        return
    mat2 = [[1, 1], [1, 0]]
    power(mat1, n // 2)
    multiply(mat1, mat1)
    if n % 2 != 0:
        multiply(mat1, mat2)

# computes the n-th fibonacci number using matrix exponentiation.
def matrix_power(n):
    if n <= 1:
        return n
    mat1 = [[1, 1], [1, 0]]
    power(mat1, n - 1)
    return mat1[0][0]
# -------------------------------------------------------------------------------


# binet's formula method
# -------------------------------------------------------------------------------
sqrt_5 = Decimal(5) ** Decimal(0.5)
phi = (Decimal(1) + sqrt_5) / Decimal(2)
phi_hat = (Decimal(1) - sqrt_5) / Decimal(2)


# uses binet's formula to compute the n-th fibonacci number.
def binet_formula(n):
    n = Decimal(n)
    return int((phi ** n - phi_hat ** n) / sqrt_5)
# -------------------------------------------------------------------------------


# memoization method
# -------------------------------------------------------------------------------
cache = {0: 0, 1: 1}


# uses memoization to store results and calculate fibonacci.
def memoization(n):
    if n in cache:
        return cache[n]
    cache[n] = memoization(n - 1) + memoization(n - 2)
    return cache[n]
# -------------------------------------------------------------------------------


# iterative method
# -------------------------------------------------------------------------------
# iterative approach to calculate the n-th fibonacci number.
def iterative(n):
    if n in {0, 1}:
        return n
    previous, fib_number = 0, 1
    for _ in range(2, n + 1):
        previous, fib_number = fib_number, previous + fib_number
    return fib_number
# -------------------------------------------------------------------------------


# fast doubling method
# -------------------------------------------------------------------------------
# uses the fast doubling algorithm to calculate fibonacci.
def fast_doubling(n):
    return _fib(n)[0]


# helper function for fast doubling.
def _fib(n):
    if n == 0:
        return (0, 1)
    else:
        a, b = _fib(n // 2)
        c = a * (b * 2 - a)
        d = a * a + b * b
        if n % 2 == 0:
            return (c, d)
        else:
            return (d, c + d)
# -------------------------------------------------------------------------------


# function to measure the time of each method
# -------------------------------------------------------------------------------
def measure_times(n_values, method):
    times = []
    for n in n_values:
        start_time = time.time()
        method(n)
        end_time = time.time()
        times.append(end_time - start_time)
    return times
# -------------------------------------------------------------------------------


# functions to create table
# -------------------------------------------------------------------------------
def create_table(n_values, recursive_times, dynamic_times, matrix_times,
                 binet_times, memoization_times, iterative_times,
                 fast_doubling_times):
    table_data = [
        ["n"] + [n for n in n_values],
        ["Recursive"] + [f"{time:.6f}" for time in recursive_times],
        ["Dynamic"] + [f"{time:.5f}" for time in dynamic_times],
        ["Matrix"] + [f"{time:.5f}" for time in matrix_times],
        ["Binet"] + [f"{time:.5f}\u200B" for time in binet_times],
        ["Memo."] + [f"{time:.5f}" for time in memoization_times],
        ["Iter."] + [f"{time:.5f}" for time in iterative_times],
        ["Doubling"] + [f"{time:.5f}" for time in fast_doubling_times]
    ]

    print(tabulate(table_data, tablefmt="grid", stralign="center",
                   numalign="center"))


def create_table_without_recursive(n_values, dynamic_times, matrix_times,
                                   binet_times, memoization_times,
                                   iterative_times, fast_doubling_times):
    table_data = [
        ["n"] + [n for n in n_values],
        ["Dynamic"] + [f"{time:.5f}" for time in dynamic_times],
        ["Matrix"] + [f"{time:.5f}" for time in matrix_times],
        ["Binet"] + [f"{time:.5f}\u200B" for time in binet_times],
        ["Memo."] + [f"{time:.5f}" for time in memoization_times],
        ["Iter."] + [f"{time:.5f}" for time in iterative_times],
        ["Doubling"] + [f"{time:.5f}" for time in fast_doubling_times]
    ]

    print(tabulate(table_data, tablefmt="grid", stralign="center",
                   numalign="center"))
# -------------------------------------------------------------------------------


# functions to plot
# -------------------------------------------------------------------------------
def plot_fibonacci_times_without_recursive(n_values, dynamic_times,
                                           matrix_times, binet_times,
                                           memoization_times, iterative_times,
                                           fast_doubling_times):
    plt.figure(figsize=(10, 6))  # Make the plot bigger
    plt.plot(n_values, dynamic_times, marker='o', label='Dynamic Programming')
    plt.plot(n_values, matrix_times, marker='o', label='Matrix Exponentiation')
    plt.plot(n_values, binet_times, marker='o', label='Binet Formula')
    plt.plot(n_values, memoization_times, marker='o', label='Memoization')
    plt.plot(n_values, iterative_times, marker='o', label='Iterative')
    plt.plot(n_values, fast_doubling_times, marker='o', label='Fast Doubling')

    plt.xlabel('n')
    plt.ylabel('time (s)')
    plt.title('time to compute nth fibonacci number')

    plt.grid(True, which='both')
    plt.minorticks_on()

    # position the legend outside to the right of the graph
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    plt.show()


def plot_fibonacci_times(n_values, recursive_times, dynamic_times, matrix_times,
                         binet_times, memoization_times, iterative_times,
                         fast_doubling_times):
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, recursive_times, marker='o', label='Recursive')
    plt.plot(n_values, dynamic_times, marker='o', label='Dynamic Programming')
    plt.plot(n_values, matrix_times, marker='o', label='Matrix Exponentiation')
    plt.plot(n_values, binet_times, marker='o', label='Binet Formula')
    plt.plot(n_values, memoization_times, marker='o', label='Memoization')
    plt.plot(n_values, iterative_times, marker='o', label='Iterative')
    plt.plot(n_values, fast_doubling_times, marker='o', label='Fast Doubling')

    plt.xlabel('n')
    plt.ylabel('time (s)')
    plt.title('time to compute nth fibonacci number')

    plt.grid(True, which='both')
    plt.minorticks_on()

    # legend outside to the right of the graph
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    plt.show()


# show only one method
def plot_one_fibonacci_times(n_values, times, label):
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, times, marker='o', label=label)
    plt.xlabel('n')
    plt.ylabel('time (s)')
    plt.title('time to compute nth fibonacci number')

    plt.grid(True, which='both')
    plt.minorticks_on()

    # legend outside to the right of the graph
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    plt.show()
# -------------------------------------------------------------------------------


# n_values = [5, 7, 10, 12, 15, 17, 20,22, 25, 27, 30, 32, 35, 37, 40, 42, 45]
n_values = [501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849]

# recursive_times = measure_times(n_values, recursive)
dynamic_times = measure_times(n_values, dynamic_programming)
matrix_times = measure_times(n_values, matrix_power)
binet_times = measure_times(n_values, binet_formula)
memoization_times = measure_times(n_values, memoization)
iterative_times = measure_times(n_values, iterative)
fast_doubling_times = measure_times(n_values, fast_doubling)


create_table_without_recursive(n_values, dynamic_times, matrix_times,
                               binet_times, memoization_times, iterative_times,
                               fast_doubling_times)

plot_fibonacci_times_without_recursive(n_values, dynamic_times, matrix_times,
                               binet_times, memoization_times, iterative_times,
                               fast_doubling_times)

# plot_one_fibonacci_times(n_values, recursive_times, 'Recursive Method')
# plot_one_fibonacci_times(n_values,dynamic_times, 'Dynamic Method')
# plot_one_fibonacci_times(n_values, fast_doubling_times,'Fast Doubling Method')
# plot_one_fibonacci_times(n_values, matrix_times, 'Matrix Power Method')
# plot_one_fibonacci_times(n_values, binet_times, 'Binet Formula Method')
# plot_one_fibonacci_times(n_values, memoization_times, 'Memoization Method')
# plot_one_fibonacci_times(n_values, iterative_times, 'Iterative Method')

