from execution import *
veca = numbers_threads
maxvec = [max(numbers_threads)]
#maxvec.append(max(numbers_threads))
print(f'veca = ', veca)
print(f'maxvec = {maxvec}')

for i in range(3):
    bs = ['gpu', 'cpu']
    for b in bs:
        if b == 'gpu':
            numbers_threads = maxvec
            print(b)
            print(f'numbers_threads = ', numbers_threads)
            print(f'veca = ', veca)
        elif b == 'cpu':
            numbers_threads = veca
            print(b)
            print(f'numbers_threads = ', numbers_threads)
            print(f'veca = ', veca)
        for n in numbers_threads:
            print(f'n = {n}')
