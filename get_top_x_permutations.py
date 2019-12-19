import itertools
import random

import numpy as np

from scipy.spatial.distance import hamming


# Build list of all possible permutations
permuts_list = list(itertools.permutations(range(9)))
permuts_array = np.array(permuts_list)
no_permuts = len(permuts_list)


# Take top x permutations which have max average hamming distance
permuts_to_take = 200
set_of_taken = set()
cnt_iterations = 0
while True:
    cnt_iterations += 1
    x = random.randint(1, no_permuts - 1)
    y = random.randint(1, no_permuts - 1)
    permut_1 = permuts_array[x]
    permut_2 = permuts_array[y]
    hd = hamming(permut_1, permut_2)

    if hd > 0.9 and (not x in set_of_taken) and (not y in set_of_taken):
        set_of_taken.add(x)
        set_of_taken.add(y)

        if len(set_of_taken) == permuts_to_take:
            break

    if cnt_iterations % 100 == 0:
        print ("Already performed count of iterations with pairs of jigsaw permutations", cnt_iterations)
        print ("Length of set of taken: ",len(set_of_taken))

print ("No of iterations it took to build top - {} permutations array = {}".format(permuts_to_take, cnt_iterations))
print ("No of permutations", len(set_of_taken))


# Build the array for selected permutation indices above
selected_permuts = []
for ind, perm_id in enumerate(set_of_taken):
    if ind < 10:
        print ("Sample permutation {}".format(ind))
        print (permuts_array[perm_id])
    selected_permuts.append(permuts_array[perm_id])

selected_permuts = np.array(selected_permuts)
np.save('selected_permuts.npy', selected_permuts)
