from cenotaph.combinatorics.transformation_groups import *
from cenotaph.combinatorics.necklaces_and_bracelets import *
from cenotaph.basics.generic_functions import convert_base

#Test base conversion
data_in = np.arange(32)
#print('Input data (decimal): {}'.format(data_in))
new_base = 2
words = convert_base(data_in, new_base)
#print('Data converted to base {}'.format(new_base))
#for i in range(len(data_in)):
    #print('{} -> {}'.format(data_in[i], data_out[i,:]))
    
##Define the groups
group_type = 'S'
method = 'invariants'
exclude_columns = {0}
representatives, labels, num_orbits = find_orbits(words, 
                                                  group_type,
                                                  method)
for i in range(words.shape[0]):
    print('{} -> {} ({})'.format(words[i,:],
                                 representatives[i,:],
                                 labels[i]))

###Find the orbits
##representatives, labels, num_orbits = find_orbits(data_out, cg, exclude = 1)
#n = 6
#k = 4
#full = True 
#allow_turnover = True
#nckls, dec_labels, inv_labels, num_orbits = necklaces(n, k, full, 
                                                      #allow_turnover)
#if full:
    #ntype = 'full'
#else:
    #ntype = 'peri'
    
#if allow_turnover:
    #ttype = 'Bracelets'
#else:
    #ttype = 'Necklaces'


#print('{} with {} beads and {} colours ({})'.format(ttype, n, k, ntype))
#print('There are {} orbits'.format(num_orbits))
#for i in range(nckls.shape[0]):
    #print('{} (dcode = {}\torbit = {})'.
          #format(nckls[i,:], dec_labels[i], inv_labels[i]))
    
#n = 8
#k = 3
#blobs, dec_labels, inv_labels, num_orbits = blobs(n, k)

#print('Blobs with {} beads and {} colours'.format(n, k))
#print('There are {} orbits'.format(num_orbits))
#for i in range(blobs.shape[0]):
    #print('{} (dcode = {}\torbit = {})'.
          #format(blobs[i,:], dec_labels[i], inv_labels[i]))


##Define the original data
#data_in = np.array([['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'H']])

##Define the groups
#cg = CyclicGroup(data_in.shape[1])
#dg = DihedralGroup(data_in.shape[1])

##Let the groups act on the original data
#data_out = cg.act_on(data_in)
#print('Effect of the {}'.format(cg))
#for i in range(data_out.shape[2]):    
    #print(data_out[:,:,i])
#print()

#data_out = dg.act_on(data_in)
#print('Effect of the {}'.format(dg))
#for i in range(data_out.shape[2]):    
    #print(data_out[:,:,i])