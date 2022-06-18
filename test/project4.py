"""Scientific Computation Project 4
Your CID here:
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



def WTdist(M,Dh):
    """
    Question 1:
    Compute WT distance matrix, X, given probability matrix, M, and Dh = D^{-1/2}
    """
    #Add code here
    X = np.zeros([len(M), len(M)])
    for i in range(len(M)):
        for j in range(i+1, len(M)):
            temp = np.dot(Dh, (M[i]-M[j]))
            X[i, j] = np.sqrt(np.dot(temp, temp))
            X[j, i] = X[i, j]
    return X


def WTdist2(M, Dh, Clist):
    """
    Question 2:
    Compute squared distance matrix, Y, given probability matrix, M, Dh = D^{-1/2},
    and community list, Clist
    """
    R = np.zeros([len(Clist), len(M)])
    Y = np.zeros([len(Clist), len(Clist)])

    # Pick the cth community
    for c in range(len(Clist)):
        # Iterate through every node
        for j in range(len(M)):
            # Every node in the community
            for i in Clist[c]:
                R[c, j] += M[i, j]/len(Clist[c])

    # Pick community a and b
    for a in range(len(Clist)):
        for b in range(a, len(Clist)):
            temp = np.dot(Dh, (R[a]-R[b]))
            Y[a, b] = np.dot(temp, temp)
            Y[b, a] = Y[a, b]

    return Y

def makeCdict(G,Clist):
    """
    For each distinct pair of communities a,b determine if there is at least one link
    between the communities. If there is, then b in Cdict[a] = a in Cdict[b] = True
    """
    m = len(Clist)
    Cdict = {}
    for a in range(m-1):
        for b in range(a+1,m):
            if len(list(nx.edge_boundary(G,Clist[a],Clist[b])))>0:
                if a in Cdict:
                    Cdict[a].add(b)
                else:
                    Cdict[a] = {b}

                if b in Cdict:
                    Cdict[b].add(a)
                else:
                    Cdict[b] = {a}
    return Cdict

def analyzeFD():
    """
    Question 4:
    Add input/output as needed

    """
    label = np.load('label4.npy')
    result = main(8)
    accuracy = 0
    pred = np.zeros(len(label))
    # Two groups
    for i in range(2):
        for j in range(len(result[i])):
            if i == 0:
                pred[result[0][j]] = 1
            else:
                pred[result[1][j]] = -1
    print(pred)
    # Compare
    for i in range(len(label)):
        if label[i] == pred[i]:
            accuracy += 1/len(pred)

    print("The accuracy is: ", accuracy)

    A = np.load('data4.npy')
    G = nx.from_numpy_array(A)
    # Plot the result
    nx.draw(G, with_labels=True, node_color=pred)
    plt.show()

    return None #modify as needed

def main(t=6,Nc=2):
    """
    WT community detection method
    t: number of random walk steps
    Nc: number of communities at which to terminate the method
    """

    #Read in graph
    A = np.load('data4.npy')
    G = nx.from_numpy_array(A)
    N = G.number_of_nodes()

    #Construct array of node degrees and degree matrices
    k = np.array(list(nx.degree(G)))[:,1]
    D = np.diag(k)
    Dinv = np.diag(1/k)
    Dh = np.diag(np.sqrt(1/k))
    label = np.load('label4.npy')

    P = Dinv.dot(A) #transition matrix
    M = np.linalg.matrix_power(P,t) #P^t
    # X = WTdist(M,Dh) #Q1: function to be completed


    #Initialize community list
    Clist = []
    for i in range(N):
        Clist.append([i])

    Y = WTdist2(M,Dh,Clist) #Q2: function to be completed


    smin = 100000
    m = len(Clist) #number of communities

    Cdict = makeCdict(G,Clist)

    #make list of community sizes
    L = []
    for l in Clist:
        L.append(len(l))

    s_list = np.zeros([m, m])
    #examine distinct pairs of communities
    for a in range(m-1):
        la = L[a]
        for b in range(a+1,m):
            lb = L[b]
            s = la * lb / (N * (la + lb)) * Y[a, b]
            s_list[a, b] = s
            s_list[b, a] = s
            if a in Cdict:
                if b in Cdict[a]:
                    if s<smin:
                        amin = a
                        bmin = b
                        smin = s

    #Q3: Add code here, use/modify/remove code above as needed
    # Merge when the cluster number is larger than the required one
    while max(L) < N/Nc:
        print(max(L))
        # Merge amin and bmin in an ordered way
        Clist[amin].extend(Clist[bmin])
        Clist[amin].sort()

        L[amin] = L[amin]+L[bmin]

        # Update the cost matrices
        for j in range(len(s_list[amin])):
            s_list[amin, j] = ((L[amin]+L[j])*s_list[amin, j]+L[j]*(s_list[bmin, j]-s_list[amin, bmin]))/(L[amin]+L[j])
            s_list[j, amin] = s_list[amin, j]

        # Update the lists
        del L[bmin], Clist[bmin]
        s_list = np.delete(s_list, bmin, axis=0)
        s_list = np.delete(s_list, bmin, axis=1)

        # Update the Cdict
        Cdict = makeCdict(G,Clist)
        smin = 100000
        m -= 1

        # examine distinct pairs of communities
        for a in range(m - 1):
            for b in range(a + 1, m):
                if a in Cdict:
                    if b in Cdict[a]:
                        if s_list[a, b] < smin:
                            amin = a
                            bmin = b
                            smin = s_list[a, b]

        print(Clist)
    # Join all other nodes into one cluster
    index = L.index(max(L))
    cluster = []
    for i in range(len(Clist)):
        if i != index:
            cluster.extend(Clist[i])

    Lfinal = [Clist[index], cluster] #modify
    return Lfinal


if __name__=='__main__':
    t=8
    out = analyzeFD()
