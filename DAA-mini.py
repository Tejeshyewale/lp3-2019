
import threading
import time

# Matrix multiplication without threads
def normal_multiply(A, B):
    rows = len(A)
    cols = len(B[0])
    result = [[0 for j in range(cols)] for i in range(rows)]

    for i in range(rows):
        for j in range(cols):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result


# Function to multiply one row using a thread
def multiply_row(A, B, result, row):
    for j in range(len(B[0])):
        for k in range(len(B)):
            result[row][j] += A[row][k] * B[k][j]


# Matrix multiplication using threads
def threaded_multiply(A, B):
    rows = len(A)
    cols = len(B[0])

    result = [[0 for j in range(cols)] for i in range(rows)]
    threads = []

    for i in range(rows):
        t = threading.Thread(
            target=multiply_row,
            args=(A, B, result, i)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return result


# Main program
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

# Normal multiplication
start = time.time()
normal_result = normal_multiply(A, B)
normal_time = time.time() - start

# Multithreaded multiplication
start = time.time()
thread_result = threaded_multiply(A, B)
thread_time = time.time() - start

# Display results
print("Matrix A:")
for row in A:
    print(row)

print("Matrix B:")
for row in B:
    print(row)

print("\nNormal Multiplication Result:")
for row in normal_result:
    print(row)

print("\nMultithreaded Result:")
for row in thread_result:
    print(row)

print("\nPerformance Comparison:")
print("Normal Time:", normal_time)
print("Threaded Time:", thread_time)
