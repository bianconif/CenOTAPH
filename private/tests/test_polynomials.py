from cenotaph.combinatorics.necklaces_and_bracelets import generate_words
from cenotaph.combinatorics import polynomials as poly

X, _ = generate_words(word_length = 4, alphabet_size = 3)

#sym_polys = poly.symmetric_polynomials(X)

#for i in range(X.shape[0]):
    #print('{} -> '.format(X[i,:]), end = '')
    #for j in range(X.shape[1] - 1):
        #print('{}'.format(sym_polys[i,j]), end = ',')
    #else:
        #print('{}'.format(sym_polys[i,-1]))
        
vdm_polys = poly.vandermonde_polynomials(X)

for i in range(X.shape[0]):
    print('{} -> {}'.format(X[i,:], vdm_polys[i]))


#P1 = poly.symmetric_polynomial(X, 1)
#P2 = poly.symmetric_polynomial(X, 2)
#print(P1)
#print(P2)

