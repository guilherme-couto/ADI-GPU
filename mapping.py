import numpy as np
import math

L = 2
deltax = 0.02
N = round(L / deltax) + 1
blockDim = 16 # x, y # Maximum of 1024 threads per block
blockZ = 1024 // (blockDim * blockDim) # z
gridDim = math.ceil(pow(N * N * N / (blockDim * blockDim * blockZ), 1 / 3)) * 3 # x, y, z

def mapping(alpha, beta, gamma):
    return alpha + beta*N + gamma*N*N

blockIdx_x = np.arange(gridDim)
blockIdx_y = np.arange(gridDim)
blockIdx_z = np.arange(gridDim)

threadIdx_x = np.arange(blockDim)
threadIdx_y = np.arange(blockDim)
threadIdx_z = np.arange(blockZ)

# a = ix | b = iy | c = iz
abc = [] # a+b*N+c*N*N
acb = [] # a+c*N+b*N*N
bac = [] # b+a*N+c*N*N
bca = [] # b+c*N+a*N*N
cab = [] # c+a*N+b*N*N
cba = [] # c+b*N+a*N*N

print("Bx\tBy\tBz\tTx\tTy\t\tix\tiy\tiz\t\tabc\tacb\tbac\tbca\tcab\tcba")
print("----------------------------------------------------------------------------------------------------------------------------------------------")
for i in blockIdx_x:
    for j in blockIdx_y:
        for m in blockIdx_z:
            for k in threadIdx_x:
                for l in threadIdx_y:
                    for n in threadIdx_z:
                        ix = i * blockDim + k
                        iy = j * blockDim + l
                        iz = m * blockZ + n
                        #print(f"{i}\t{j}\t{m}\t{k}\t{l}\t\t{ix}\t{iy}\t{iz}\t\t{mapping(ix, iy, iz)}\t{mapping(ix, iz, iy)}\t{mapping(iy, ix, iz)}\t{mapping(iy, iz, ix)}\t{mapping(iz, ix, iy)}\t{mapping(iz, iy, ix)}")
                        if ix < N and iy < N and iz < N:
                            abc.append(mapping(ix, iy, iz))
                            acb.append(mapping(ix, iz, iy))
                            bac.append(mapping(iy, ix, iz))
                            bca.append(mapping(iy, iz, ix))
                            cab.append(mapping(iz, ix, iy))
                            cba.append(mapping(iz, iy, ix))
                        

print("\n")
abc.sort()
acb.sort()
bac.sort()
bca.sort()
cab.sort()
cba.sort()

print(f"N = {N}")
print(f"Total number of elements = {N * N * N}")
print("\n")

print(f"Grid (x, y, z) = ({gridDim}, {gridDim}, {gridDim}) -> {gridDim * gridDim * gridDim} blocks per grid")
print(f"Block (x, y, z) = ({blockDim}, {blockDim}, {blockZ}) -> {blockDim * blockDim * blockZ} threads per block")
print(f"Total number of threads = {gridDim * gridDim * gridDim * blockDim * blockDim * blockZ}")
print("\n")

#print(f"abc (ix + iy*N + iz*N*N) = {abc}")
print(f"len(abc) (ix + iy*N + iz*N*N) = {len(abc)}")
print("\n")

#print(f"acb (ix + iz*N + iy*N*N) = {acb}")
print(f"len(acb) (ix + iz*N + iy*N*N) = {len(acb)}")
print("\n")

#print(f"bac (iy + ia*N + iz*N*N) = {bac}")
print(f"len(bac) (iy + ia*N + iz*N*N) = {len(bac)}")
print("\n")

#print(f"bca (iy + iz*N + ix*N*N) = {bca}")
print(f"len(bca) (iy + iz*N + ix*N*N) = {len(bca)}")
print("\n")

#print(f"cab (iz + ix*N + iy*N*N) = {cab}")
print(f"len(cab) (iz + ix*N + iy*N*N) = {len(cab)}")
print("\n")

#print(f"cba (iz + iy*N + ix*N*N) = {cba}")
print(f"len(cba) (iz + iy*N + ix*N*N) = {len(cba)}")
print("\n")


# Check if there are repeated elements
for i in range(len(abc) - 1):
    if abc[i] == abc[i + 1]:
        print(f"Repeated element in abc: {abc[i]}")
        break
    if acb[i] == acb[i + 1]:
        print(f"Repeated element in acb: {acb[i]}")
        break
    if bac[i] == bac[i + 1]:
        print(f"Repeated element in bac: {bac[i]}")
        break
    if bca[i] == bca[i + 1]:
        print(f"Repeated element in bca: {bca[i]}")
        break
    if cab[i] == cab[i + 1]:
        print(f"Repeated element in cab: {cab[i]}")
        break
    if cba[i] == cba[i + 1]:
        print(f"Repeated element in cba: {cba[i]}")
        break
    
    # Check if the difference between elements is 1
    if abc[i + 1] - abc[i] != 1:
        print(f"Missing element in abc: {abc[i] + 1}")
        break
    if acb[i + 1] - acb[i] != 1:
        print(f"Missing element in acb: {acb[i] + 1}")
        break
    if bac[i + 1] - bac[i] != 1:
        print(f"Missing element in bac: {bac[i] + 1}")
        break
    if bca[i + 1] - bca[i] != 1:
        print(f"Missing element in bca: {bca[i] + 1}")
        break
    if cab[i + 1] - cab[i] != 1:
        print(f"Missing element in cab: {cab[i] + 1}")
        break
    if cba[i + 1] - cba[i] != 1:
        print(f"Missing element in cba: {cba[i] + 1}")
        break


