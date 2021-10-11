"""Basic functions for combinatorics of word, necklaces and bracelets"""
import numpy as np

from cenotaph.basics.generic_functions import convert_base
import cenotaph.combinatorics.transformation_groups as tg
from cenotaph.combinatorics.polynomials import symmetric_polynomials,\
     vandermonde_polynomial

def generate_words(word_length = 8, alphabet_size = 2):
    """Generate all the possible words of given length with letters drawn from 
    an alphabet of given size
    
    Parameters
    ----------
    word_length : int
        The length of the words
    alphabet_size : int
        The number of different symbols in the alphabet
        
    Returns
    -------
    words : ndarray of int (alphabet_size ** word_length, word_length)
        All the words that can be generated. Each word is given as a sequence
        of integers from 0 to (alphabet_size - 1)
    decimal_codes : ndarray of int (alphabet_size ** word_length)
        The decimal label of each word
    """
    
    #Generate the decimal codes of the words
    decimal_codes = np.arange(alphabet_size ** word_length)
    
    #Get string representation in base alphabet_size
    words = convert_base(decimal_codes, alphabet_size)
    return words, decimal_codes

def generate_lyndon_words(word_length = 8, alphabet_size = 2):
    """Generate all the possible Lyndon words of given length with letters 
    drawn from an alphabet of given size
    
    Parameters
    ----------
    word_length : int
        The length of the words
    alphabet_size : int
        The number of different symbols in the alphabet
        
    Returns
    -------
    lyndon_words : ndarray of int (alphabet_size ** word_length, word_length)
        All the Lyndon words that can be generated. Each word is given as a 
        sequence of integers from 0 to (alphabet_size - 1)
        
    Credits
    -------
    Sourced from: https://gist.github.com/dvberkel/1950267
    """   
    
    def yield_length_limited_lyndon_words(s,n):
        """Generate nonempty Lyndon words of length <= n over an s-symbol alphabet.
        The words are generated in lexicographic order, using an algorithm from
        J.-P. Duval, Theor. Comput. Sci. 1988, doi:10.1016/0304-3975(88)90113-2.
        As shown by Berstel and Pocchiola, it takes constant average time
        per generated word."""
        w = [-1]                            # set up for first increment
        while w:
            w[-1] += 1                      # increment the last non-z symbol
            yield w
            m = len(w)
            while len(w) < n:               # repeat word to fill exactly n syms
                w.append(w[-m])
            while w and w[-1] == s - 1:     # delete trailing z's
                w.pop()    

    def yield_lyndon_words_with_length(s,n):
        """Generate Lyndon words of length exactly n over an s-symbol alphabet.
        Since nearly half of the outputs of LengthLimitedLyndonWords(s,n)
        have the desired length, it again takes constant average time per word."""   
        if n == 0:
            yield []    # the empty word is a special case not handled by main alg
        for w in yield_length_limited_lyndon_words(s,n):
            if len(w) == n:
                yield w     
    
    lyndon_words = np.zeros((0,word_length), dtype = np.int)
    lyndon_words_generator = yield_lyndon_words_with_length(s = alphabet_size, 
                                                            n = word_length)
    
    for lyndon_word in lyndon_words_generator:
        lyndon_words = np.vstack((lyndon_words, lyndon_word))
        
    #Remove the duplicates
    lyndon_words = np.unique(ar = lyndon_words, axis = 0)    
    
    return lyndon_words
    
   

def compute_word_period(words):
    """Computes the word period. This is the length of the minimum subword 
    which enables the original word to written as the concatenation of two
    or more copies of the subword
    
    Parameters
    ----------
    words : nparray of int (W,N)
        A two-dimensional array where each row represents an N-word
        
    Returns
    -------
    periods : nparray of int (W)
        The priod of each word
        
    References
    ----------
    [1] Archer, C.; Lauderdale, L.-K. Enumeration of cyclic permutations in vector 
        grid classes Journal of Combinatorics 11(1):203-230
    """
    
    #Set the dafult period as the length of the words (aperiodic words)
    periods = np.ones((words.shape), dtype = np.int) * words.shape[1]
    
    for i in range(words.shape[1]):
        shifted = np.roll(words, shift = i + 1, axis = 1)
        comparisons = np.all(a = (shifted == words), axis = 1)
        periods[np.where(comparisons), i] = i + 1
    
    periods = np.min(a = periods, axis = 1)
    return periods
        

def find_orbits(words, group_type, method, **kwargs):
    """ Find orbits
    
    Parameters
    ----------
    words : ndarray of int (M,N)
        The input data, where each row represents one word. The
        words need to have the same length. The character of each word should
        be an integer in the range [0,...,alphabet_size - 1]; the maximum
        value of {words} defines the size of the alphabet - i.e.:
        alphabet_size = max(words[:]) + 1.
    group_type : str
        A string indicating the transformation group acting on words. Can be:
            'A' -> Alternating group
            'C' -> Cyclic group
            'D' -> Dihedral group
            'S' -> Symmetric group
    method : str 
        The algorithm used for computing the orbits. Can be:
            'brute-force' -> see _find_orbits_brute_force()
            'invariants'  -> see _find_orbits_invariants()
        See related functions fro a description.
    exclude : list of int (optional)
        The indices of the characters that will not be affected by the 
        transformation group. For instance, let
        words = [1, 2, 3, 4], transformation_group = CyclicGroup and
        exclude = 1. In this case the action of transformation_group on words 
        will produce the following (exluding the identity): [1, 4, 2, 3]; 
        [1, 3, 4, 2] and [1, 2, 3, 4]. 
        
    Returns
    -------
    representatives : ndarray of int (M,N)
        Each row contains the representative of the orbit the corresponding 
        word belongs to.
    inv_labels : ndarray of int (M)
        Each row contains the invariant label (i.e. orbit number, in decimal 
        format) of the corresponding word.
    num_orbits : int
        The total number of orbits.
    """
    
    representatives, inv_labels, num_orbits = None, None, None
    
    #Compute the group order
    group_order = words.shape[1]
    if 'exclude' in kwargs:
        group_order = group_order - len(kwargs['exclude'])
    
    #Select method to compute the orbits
    if method == 'brute-force':
        group = tg.TransformationGroup.factory(group_type, 
                                               group_order,
                                               include_id = True)
        representatives, inv_labels, num_orbits =\
            _find_orbits_brute_force(words, group, **kwargs)
    elif method == 'invariants':
        representatives, inv_labels, num_orbits =\
            _find_orbits_via_invariants(words, group_type, **kwargs)
    else:
        raise Exception('Method for finding orbits unknown')
    
    return representatives, inv_labels, num_orbits

def compute_transitions(words):
    """Compute the number of transitions in words. A transition occurs
    whenever two adjacent letters in a word are different. The words are
    considered as cyclic string, therefore the first and last letter of
    a word are also compared abd, if different, one transition is counted.
    
    Parameters
    ----------
    words : ndarray of int (M,N)
        The input data, where each row represents one word. The
        words need to have the same length. The character of each word should
        be an integer in the range [0,...,alphabet_size - 1]; the maximum
        value of {words} defines the size of the alphabet - i.e.:
        alphabet_size = max(words[:]) + 1. 
        
    Returns
    -------
    transitions : ndarray of int (M)
        The number of transition in each word
    """
    
    transitions = np.zeros(shape = (words.shape[0]), dtype = np.int), 
    
    for i in range(-1, words.shape[1] - 1):
        comparisons = np.array(words[:,i] != words[:,i + 1], dtype = np.int)
        transitions = transitions + comparisons
        
    return transitions
        
    
def _find_orbits_via_invariants(words, group_type, **kwargs):
    """Determine the orbits by computing a minimal set of invariants. 
        Parameters
    ----------
    words : ndarray of int (M,N)
        The input data, where each row represents one word. The
        words need to have the same length. The character of each word should
        be an integer in the range [0,...,alphabet_size - 1]; the maximum
        value of {words} defines the size of the alphabet - i.e.:
        alphabet_size = max(words[:]) + 1.
    group_type : str
        A string indicating the transformation group acting on the set of 
        words. Accepted values are:
            'A' -> slternating group
            'S' -> symmetric group
    exclude : set of int (optional)
        The indices of the characters that will not be affected by the 
        transformation group. For instance, let
        words = [1, 2, 3, 4], transformation_group = CyclicGroup and
        exclude = 1. In this case the action of transformation_group on words 
        will produce the following (exluding the identity): [1, 4, 2, 3]; 
        [1, 3, 4, 2] and [1, 2, 3, 4].
        
    Returns
    -------
    representatives : ndarray of int (M,N)
        Each row contains the representative of the orbit the corresponding 
        word belongs to.
    inv_labels : ndarray of int (M)
        Each row contains the invariant label (i.e. orbit number, in decimal 
        format) of the corresponding word.
    num_orbits : int
        The total number of orbits.
    """
    
    exclude_columns = None
    if 'exclude' in kwargs:
        exclude_columns = kwargs['exclude']
    
    if exclude_columns:
        include_columns = set(range(words.shape[1])) - exclude_columns
        to_be_transformed = words[:,list(include_columns)]
    else:
        to_be_transformed = words
    
    #Compute the invariants
    if group_type == 'A':
        #Invariants are the symmetric polynomials plus the Vandermonde
        #polynomial
        invariants = symmetric_polynomials(to_be_transformed)
        invariants = np.hstack((invariants, 
                                vandermonde_polynomial(to_be_transformed)))
    elif group_type == 'S':
        #Invariants are the symmetric polynomials
        invariants = symmetric_polynomials(to_be_transformed)
    else:
        raise Exception('Group tyope not supported')
    
    ##Concatenate the invariants to the left
    #transformed_words = np.hstack((words, invariants))
    
    ##Mark with -1 the excluded positions
    #if exclude_columns:
        #fillings = -1 * np.ones((words.shape[0],len(include_columns)))
        #transformed_words[:,list(include_columns)] = fillings
    
    #Compute the orbits
    _, idx, inv_labels = np.unique(invariants,
                                   axis = 0,
                                   return_index = True,
                                   return_inverse=True)
    unique_labels = set(inv_labels)
    num_orbits = len(unique_labels)
    representatives_idxs = [list(inv_labels).index(label)\
                            for label in inv_labels]
    representatives = words[representatives_idxs, :]
    return representatives, inv_labels, num_orbits

def _find_orbits_brute_force(words, transformation_group, **kwargs):
    """Determine the orbits via brute-force approach. For each word apply
    all the group transformation and retain, as representative, the one
    with the minimum value.
    
    Parameters
    ----------
    words : ndarray of int (M,N)
        The input data, where each row represents one word. The
        words need to have the same length. The character of each word should
        be an integer in the range [0,...,alphabet_size - 1]; the maximum
        value of {words} defines the size of the alphabet - i.e.:
        alphabet_size = max(words[:]) + 1.
    transformation_group : TransformationGroup
        The transformation group acting on the set of words.
    exclude : set of int (optional)
        The indices of the characters that will not be affected by the 
        transformation group. For instance, let
        words = [1, 2, 3, 4], transformation_group = CyclicGroup and
        exclude = 1. In this case the action of transformation_group on words 
        will produce the following (exluding the identity): [1, 4, 2, 3]; 
        [1, 3, 4, 2] and [1, 2, 3, 4].
        
    Returns
    -------
    representatives : ndarray of int (M,N)
        Each row contains the representative of the orbit the corresponding 
        word belongs to.
    inv_labels : ndarray of int (M)
        Each row contains the invariant label (i.e. orbit number, in decimal 
        format) of the corresponding word.
    num_orbits : int
        The total number of orbits.
    """
    
    exclude_columns = None
    if 'exclude' in kwargs:
        exclude_columns = kwargs['exclude']
    
    #Determine the size of the alphabet
    alphabet_size = np.max(words[:]) + 1
    
    #Compute the transformed words
    if not exclude_columns:
        transformed_words = transformation_group.act_on(words)
    else:
        #Let the group act on the selected part 
        include_columns = set(range(words.shape[1])) - exclude_columns
        to_be_transformed = words[:,list(include_columns)]
        transformed_part = transformation_group.act_on(to_be_transformed)
        
        #Now deal with the part that needs to stay put
        untransformed_part = words[:,list(exclude_columns)]
        untransformed_part = np.expand_dims(untransformed_part, axis = -1)
        untransformed_part = np.repeat(untransformed_part, 
                                       transformed_part.shape[2], 
                                       axis = 2)
        
        #Merge the two parts together
        out_shape = (words.shape[0],
                     transformed_part.shape[1] + untransformed_part.shape[1],
                     transformed_part.shape[2])
        transformed_words = np.zeros(out_shape, dtype = words.dtype)
        transformed_words[:,list(include_columns),:] = transformed_part
        transformed_words[:,list(exclude_columns),:] = untransformed_part
    
    #Define the mask of weights
    weights = alphabet_size ** np.arange(words.shape[1]) 
    weights = weights[::-1] #Reverse the order
    weights = np.tile(weights, (transformed_words.shape[0], 1))
    weights = np.expand_dims(weights, axis = -1)
    weights = np.repeat(weights, transformed_words.shape[2], axis = 2)
    
    #Multiply and sum
    res = np.sum(np.multiply(transformed_words, weights), axis = 1)
    
    #Get the representatives and orbit labels as the minimum values of the
    #operation above
    idxs = np.argmin(res, axis = 1)
    representatives = np.zeros(words.shape, dtype = int)
    raw_orbit_labels = np.zeros(words.shape[0], dtype = int)
    for i in range(len(idxs)):
        representatives[i,:] = transformed_words[i,:,idxs[i]]
        raw_orbit_labels[i] = res[i,idxs[i]]
    
    inv_labels, num_orbits = _map_labels(raw_orbit_labels)
        
    return representatives, inv_labels, num_orbits

def blobs(n, k):
    """Equivalence classes of sequences of n beads and k colours under
    the action of the symmetric group Sn
    
    Parameters
    ----------
    n : int
        The total number of beads.
    k : int 
        The number of colours.
        
    Returns
    -------
    blobs : ndarray of int (M,n)
        Each row contains the representation of the m-th blob. Different 
        colours are encoded with different integers.
    dec_labels : ndarray of int (M)
        Each element contains the non-invariant label (in decimal format) 
        of the m-th blob.
    inv_labels : ndarray of int (M)
        The invariant label (in decimal format) of the m-th blob.
    num_orbits : int
        The number of intrinsically-different blobs.
    """
    
    #Generate all the words with n beads and k colours
    words, dec_labels = generate_words(word_length = n, alphabet_size = k)
    
    #Sort each sequence in descending order from left to right
    blobs = np.sort(words, axis = 1)
    
    #Compute the invariant labels
    mask = k ** np.arange(blobs.shape[1])
    mask = np.tile(mask, (blobs.shape[0], 1))
    inv_labels = np.sum(np.multiply(mask, blobs), axis = 1)    
    
    #Compute the number of orbits
    inv_labels, num_orbits = _map_labels(inv_labels)
    
    return blobs, dec_labels, inv_labels, num_orbits

def necklaces(n, k, full = False, allow_turnover = False):
    """Necklaces/bracelets with n beads and k colours
    
    Parameters
    ----------
    n : int
        The total number of beads.
    k : int 
        The number of colours.
    full : bool
        Whether the necklace has a central bead.
    allow_turnover : bool
        If True two sequences are considered equivalent if they can be
        transformed into one another via a reflection across some diameter 
        (bracelets).
        
    Returns
    -------
    necklaces : ndarray of int (M,n)
        Each row contains the representation of the m-th necklace/bracelet. 
        Different colours are encoded with different integers.
    dec_labels : ndarray of int (M)
        Each element contains the non-invariant label (in decimal format) 
        of the m-th necklace/bracelet.
    inv_labels : ndarray of int (M)
        The invariant label (in decimal format) of the m-th necklace/bracelet.
    num_orbits : int
        The number of intrinsically-different necklaces/bracelets.
        
    References
    ----------
    [1] Bianconi, F., González, E.
        Counting local n-ary patterns
        (2019) Pattern Recognition Letters, 117, pp. 24-29. 
    """
    
    #Set the correct transformation group
    if allow_turnover:
        transformation_group = 'A'
    else:
        transformation_group = 'C'
    
    if full:
        #Generate the peripheral ring
        words, _ = generate_words(word_length = n - 1, alphabet_size = k)
        num_words = words.shape[0]
        words = np.tile(words, (k, 1))
        
        #Add the central point
        central_point = np.arange(k)
        central_point = np.repeat(central_point, num_words, axis = -1)
        #central_point = np.squeeze(central_point)
        central_point = np.expand_dims(central_point, axis = -1)
        words = np.concatenate((central_point, words), axis = 1)
        
        #Let the group act on the peripheral points only
        necklaces, inv_labels, num_orbits = find_orbits(words, 
                                                        transformation_group,
                                                        exclude = 1)
    else:
        words, _ = generate_words(word_length = n, alphabet_size = k)
        necklaces, inv_labels, num_orbits = find_orbits(words, 
                                                        transformation_group,
                                                        method = 'brute-force')
    
    #Compute the non-invariant labels
    mask = k ** np.arange(words.shape[1])
    mask = np.tile(mask, (words.shape[0], 1))
    dec_labels = np.sum(np.multiply(mask, words), axis = 1)
      
    return necklaces, dec_labels, inv_labels, num_orbits

def _map_labels(labels):
    """Map the given labels to the {0,...,num_labels - 1} set, where 
    num_labels = len(labels).
    
    Parameters
    ----------
    labels : int
        The input labels
        
    Returns
    -------
    mapped_labels : int
        The mapped labels
    num_diff_labels : int
        The number of different labels
    """
    
    #Compute the labels' dictionary
    dictionary = np.unique(labels)
    num_labels = len(dictionary)

    #Define the lookup table and map the labels
    lookup_table = dict(zip(dictionary, np.arange(num_labels)))
    mapped_labels = [lookup_table[x] for x in labels]  
    
    return mapped_labels, num_labels
    