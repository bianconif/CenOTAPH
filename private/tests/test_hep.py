import numpy as np

import cenotaph.basics.neighbourhood as nb
import cenotaph.combinatorics.transformation_groups as tg
from cenotaph.texture.hep.basics import group_invariant_dictionary
from cenotaph.texture.hep.basics import replace_words

#Test replace_words()
data_in = np.array([[1,2,3,4],[1,2,4,3]])
old_dictionary = [1,2,3,4]
new_dictionary = [11,21,31,41]
data_out = replace_words(data_in, old_dictionary, new_dictionary)

neighbourhood = nb.DigitalCircleNeighbourhood(radius = 1, 
                                              max_num_points = 4, 
                                              full=True)
num_letters = 2
dictionary_in = range(num_letters ** neighbourhood.get_num_points())
dictionary_out = group_invariant_dictionary(neighbourhood, 
                                            dictionary_in,  
                                            group_action = 'C', 
                                            action_type = 'peripheral')
for _, i in enumerate(dictionary_in):
    print('{}\t{}'.format(dictionary_in[i], dictionary_out[i]))
a = 0