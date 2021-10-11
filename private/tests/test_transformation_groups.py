from cenotaph.combinatorics.transformation_groups import *
from cenotaph.combinatorics.necklaces_and_bracelets import *
from cenotaph.basics.generic_functions import convert_base

n = 6                   #Number of beads
k = 4                   #Number of colours
num_words = k ** n      #Number of words

data_in = np.arange(num_words)
data_in = convert_base(data_in, k)
    
#Define the groups
sg = SymmetricGroup(data_in.shape[1])
ag = AlternatingGroup(data_in.shape[1])
cg = CyclicGroup(data_in.shape[1])
dg = DihedralGroup(data_in.shape[1])
groups = [sg, ag, cg, dg]

for g in groups:
    representatives, labels, num_orbits = find_orbits(data_in, g)
    print('{} acting on sequences of {} beads and {} colours'
          .format(g, n, k))
    print('There are {} orbits'.format(num_orbits))
    for i in range(representatives.shape[0]):
        print('{}\t{}'.format(representatives[i,:], labels[i]))
    print()

