import random
import time
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(10 ** 6)


# Sorting Algorithms
def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            swap(arr, i, j)
    swap(arr, i + 1, high)
    return i + 1


def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)


def merge(arr, left, mid, right):
    n1, n2 = mid - left + 1, right - mid
    L, R = arr[left:left + n1], arr[mid + 1:mid + 1 + n2]

    i, j, k = 0, 0, left
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    arr[k:k + n1 - i] = L[i:n1]
    arr[k:k + n2 - j] = R[j:n2]


def merge_sort(arr, left, right):
    if left < right:
        mid = (left + right) // 2
        merge_sort(arr, left, mid)
        merge_sort(arr, mid + 1, right)
        merge(arr, left, mid, right)


def heapify(arr, n, i):
    largest, l, r = i, 2 * i + 1, 2 * i + 2
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)


def bogo_sort(a):
    n = len(a)
    while not is_sorted(a):
        shuffle(a)


def is_sorted(a):
    n = len(a)
    for i in range(n - 1):
        if a[i] > a[i + 1]:
            return False
    return True


def shuffle(a):
    n = len(a)
    for i in range(n):
        r = random.randint(0, n - 1)
        a[i], a[r] = a[r], a[i]


# Array Generation Functions
def generate_random_array(size):
    return [random.randint(-5000, 5000) for _ in range(size)]


def generate_sorted_array(size):
    return list(range(-size // 2, size // 2))


def generate_reverse_sorted_array(size):
    return list(range(size // 2, -size // 2, -1))


def generate_nearly_sorted_array(size, swaps=10):
    arr = list(range(-size // 2, size // 2))
    for _ in range(swaps):
        i, j = random.randint(0, size - 1), random.randint(0, size - 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def generate_few_unique_array(size, unique_count=10):
    return [random.randint(-unique_count, unique_count) for _ in range(size)]


def generate_all_duplicates_array(size):
    return [random.randint(-100, 100)] * size


def generate_all_positive_array(size):
    return [random.randint(1, 10000) for _ in range(size)]


def generate_all_negative_array(size):
    return [random.randint(-10000, -1) for _ in range(size)]


def generate_mixed_sign_array(size):
    return [random.randint(-5000, 5000) for _ in range(size)]


def generate_alternating_sign_array(size):
    return [(i if i % 2 == 0 else -i) for i in range(1, size + 1)]


# Timing Function
def time_sorting_algorithm(sort_function, arr):
    start_time = time.time()
    sort_function(arr)
    return time.time() - start_time


# Wrappers for Sorting Algorithms
def quick_sort_wrapper(arr):
    quick_sort(arr, 0, len(arr) - 1)


def merge_sort_wrapper(arr):
    merge_sort(arr, 0, len(arr) - 1)


def heap_sort_wrapper(arr):
    heap_sort(arr)


# Experiment Setup
sizes = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
bogo_sizes = [3, 4, 5, 6, 7]

array_types = {
    "Random": generate_random_array,
    "Sorted": generate_sorted_array,
    "Reverse Sorted": generate_reverse_sorted_array,
    "Nearly Sorted": generate_nearly_sorted_array,
    "Few Unique": generate_few_unique_array,
    "All Duplicates": generate_all_duplicates_array,
    "All Positive": generate_all_positive_array,
    "All Negative": generate_all_negative_array,
    "Mixed Sign": generate_mixed_sign_array,
    "Alternating Sign": generate_alternating_sign_array
}

bogo_array_types = {
    "Random": generate_random_array,
    "Sorted": generate_sorted_array,
    "Reverse Sorted": generate_reverse_sorted_array
}

sorting_algorithms = {
    "QuickSort": quick_sort_wrapper,
    "MergeSort": merge_sort_wrapper,
    "HeapSort": heap_sort_wrapper
}

# Results Storage
results = {alg: {arr_type: [] for arr_type in array_types} for alg in sorting_algorithms}
bogo_results = {arr_type: [] for arr_type in bogo_array_types}

# Run Experiments for Main Algorithms
for size in sizes:
    for arr_type, generator in array_types.items():
        arr = generator(size)
        for alg_name, alg_func in sorting_algorithms.items():
            arr_copy = arr[:]
            exec_time = time_sorting_algorithm(alg_func, arr_copy)
            results[alg_name][arr_type].append(exec_time)

# Generate Performance Tables for Each Sorting Algorithm
for alg_name in sorting_algorithms:
    print(f"\n{alg_name} Performance Table")
    headers = ["Array Type"] + [str(size) for size in sizes]
    max_arr_type_len = max(len(arr_type) for arr_type in array_types.keys())
    max_arr_type_len = max(max_arr_type_len, len("Array Type"))

    header_format = "  ".join(["{:<" + str(max_arr_type_len) + "}"] + ["{:>12}" for _ in sizes])
    row_format = "  ".join(["{:<" + str(max_arr_type_len) + "}"] + ["{:>12.6f}" for _ in sizes])

    print(header_format.format(*headers))
    for arr_type in array_types:
        times = results[alg_name][arr_type]
        row = [arr_type] + times
        print(row_format.format(*row))

# Plot Results for Each Sorting Algorithm
for alg_name, data in results.items():
    plt.figure()
    for arr_type, times in data.items():
        plt.plot(sizes, times, label=arr_type)
    plt.xlabel("Array Size")
    plt.ylabel("Time (seconds)")
    plt.title(f"Performance of {alg_name}")
    plt.legend()
    plt.show()

# Generate Performance Tables for Each Array Type
for arr_type in array_types:
    print(f"\n{arr_type} Performance Table")
    headers = ["Algorithm"] + [str(size) for size in sizes]
    max_alg_len = max(len(alg) for alg in sorting_algorithms.keys())
    max_alg_len = max(max_alg_len, len("Algorithm"))

    header_format = "  ".join(["{:<" + str(max_alg_len) + "}"] + ["{:>12}" for _ in sizes])
    row_format = "  ".join(["{:<" + str(max_alg_len) + "}"] + ["{:>12.6f}" for _ in sizes])

    print(header_format.format(*headers))
    for alg_name in sorting_algorithms:
        times = results[alg_name][arr_type]
        row = [alg_name] + times
        print(row_format.format(*row))

# Plot Results for Each Array Type
for arr_type in array_types:
    plt.figure()
    for alg_name in sorting_algorithms:
        times = results[alg_name][arr_type]
        plt.plot(sizes, times, label=alg_name)
    plt.xlabel("Array Size")
    plt.ylabel("Time (seconds)")
    plt.title(f"Performance on {arr_type} Arrays")
    plt.legend()
    plt.show()

# Run BogoSort Experiments
for size in bogo_sizes:
    for arr_type, generator in bogo_array_types.items():
        arr = generator(size)
        arr_copy = arr[:]
        exec_time = time_sorting_algorithm(bogo_sort, arr_copy)
        bogo_results[arr_type].append(exec_time)

# Generate BogoSort Table
print("\nBogoSort Performance Table")
headers = ["Array Type"] + [str(size) for size in bogo_sizes]
max_arr_type_len = max(len(arr_type) for arr_type in bogo_array_types.keys())
max_arr_type_len = max(max_arr_type_len, len("Array Type"))

header_format = "  ".join(["{:<" + str(max_arr_type_len) + "}"] + ["{:>12}" for _ in bogo_sizes])
row_format = "  ".join(["{:<" + str(max_arr_type_len) + "}"] + ["{:>12.6f}" for _ in bogo_sizes])

print(header_format.format(*headers))
for arr_type in bogo_array_types:
    times = bogo_results[arr_type]
    row = [arr_type] + times
    print(row_format.format(*row))

# Plot BogoSort Results
plt.figure()
for arr_type, times in bogo_results.items():
    plt.plot(bogo_sizes, times, marker='o', linestyle='-', label=arr_type)
plt.xlabel("Array Size")
plt.ylabel("Time (seconds)")
plt.title("Performance of BogoSort")
plt.legend()
plt.show()