import numpy as np
import random

'''
def column(matrix, i):
    return [row[i]] for row in matrix]
'''
def naive(probs):
    probs[6] = max(probs)
    probs[8] = max(probs[8], probs[4], probs[0], probs[3], probs[5])
    probs[1] = max(probs[1], probs[7], probs[2])
    probs[3] = max(probs[3], probs[5])
    return probs


def sigmoid(inVal):
    if inVal < 0.05:
        return 0.05
    return ( 2.0 / ( 0.8 + 700*np.exp(-12*inVal) ) ) + 0.05

def toMatrix(probs):
    return np.array([
        [0,0,0,0,0,0,0,0,sigmoid(probs[0]*probs[8])],
        [0,0,sigmoid(probs[1]*probs[2]),0,0,0,sigmoid(probs[1]*probs[6]),sigmoid(probs[1]*probs[7]),0],
        [0,sigmoid(probs[2]*probs[1]),0,0,0,0,0,0,0],
        [0,0,0,0,0,sigmoid(probs[3]*probs[5]),0,0,sigmoid(probs[3]*probs[8])],
        [0,0,0,0,0,0,0,0,sigmoid(probs[4]*probs[8])],
        [0,0,0,sigmoid(probs[5]*probs[3]),0,0,0,0,0],
        [0,sigmoid(probs[6]*probs[1]),0,0,0,0,0,0,sigmoid(probs[6]*probs[8])],
        [0,sigmoid(probs[7]*probs[1]),0,0,0,0,0,0,0],
        [sigmoid(probs[8]*probs[0]),0,0,sigmoid(probs[8]*probs[3]),sigmoid(probs[8]*probs[4]),0,sigmoid(probs[8]*probs[6]),0,0],
    ])

def toMarkov(G, getNonZero = False):
    M = G.copy()
    rowSums = M.sum(axis=1, dtype = np.float16)[:,np.newaxis] #compute row sums
    r_i = M.nonzero()[0] #get nonzero row indices
    M[r_i,:] = M[r_i,:] / rowSums[r_i] #normalize
    if getNonZero:
        return M, r_i
    return M


def pageRank(G, s = 0.85, max_err = 0.001):
    '''
    Compute pageRank

    Parameters:
        G:
            Graph matrix representing state transitions
            For weighted pageRank, G elements can be any non-negative reals

        S:
            Damping factor, probability the surfer will keep moving

        max_err:
            exit condition for the matrix power iteration, the computation
            ends when the computed difference is below this value
    '''

    num_rows = G.shape[0] #number of rows in the G matrix

    #Convert G to a markov matrix we label as M
    M = G.astype(float) #make it store floats
    rowSums = M.sum(axis=1)[:,np.newaxis] #compute row sums
    r_i = M.nonzero()[0] #get nonzero row indices
    M[r_i,:] = M[r_i,:] / rowSums[r_i] #normalize
    sink = 0 #row sums = 0?
    #compute pageRank, r
    page_0, page = np.zeros(num_rows), np.ones(num_rows)
    diff = np.sum(np.abs(page - page_0))
    while diff > max_err:
        page_0 = page.copy()
        #calculate pageRank for current iteration
        for i in range(0,num_rows):
            #inlinks of state i
            I_i  = M[:,i]
            #account for sink states
            S_i = sink / float(num_rows)
            #account for movement to state i
            T_i = np.ones(num_rows) / float(num_rows)
            #Weighted pageRank evaluation
            page[i] = page_0.dot(I_i * s + S_i * s + T_i * (1-s) * G[i])

        current = page / float(sum(page))
        diff = np.sum(np.abs(page - page_0))
        #current = current.multiply(rowSums)
    current = np.transpose(np.matrix(current))
    current[r_i] = np.multiply(current[r_i],rowSums[r_i])
    # .A1 returns the base array of the matrix
    return current.A1

if __name__ == "__main__":
    G = np.array([[1,1,0,0,0,0,0],
                  [1,1,0,0,0,0,0],
                  [0,0,1,0,0,0,0],
                  [0,0,0,1,0,0,0],
                  [0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,0],
                  [0,0,0,0,0,1,1]])

    print(pageRank(G))
    print(G)
