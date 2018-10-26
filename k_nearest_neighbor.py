import math
import numpy as np
import scipy.io
from calculate_cosine_distance import cosineDistance
import sys


def load_data(filename):
    f = scipy.io.loadmat(filename)
    traindata = f['traindata']
    trainlabels = f['trainlabels']
    testdata = f['testdata']
    evaldata = f['evaldata']
    testlabels = f['testlabels']
    return traindata, trainlabels, testdata, evaldata, testlabels


def calculate_distances(trainlabels, traindata):
  
    distances = []
    for i in range(len(trainlabels)):
        tr_ex_c1 = traindata[i]
        aux = []
        for j in range (len(trainlabels)):
             if i != j:
                 tr_ex_c2 = traindata[j]
                 d = cosineDistance(tr_ex_c1, tr_ex_c2)
                 aux.append(d)
        distances.append(aux) 
    return distances


def get_closest_k_points(D, k):
    return sorted(D)[:k+1] 


def get_index_vec(points_list, D):
    
    index_vec = []
    while points_list:
        point  = points_list.pop()
	for j, point_D in enumerate(D):
            if  np.isclose(point_D, point, rtol=1e-8, atol=1e-08, equal_nan=False) and \
                           j not in set(index_vec):
                index_vec.append(j)
                break
        
    return index_vec


def search(x, D, k):
    points_list = get_closest_k_points(D, k)
    return get_index_vec(points_list, D)


def classify(index_vec, trainlabels):
   
    label1 = trainlabels[0]
    label2 = None
    neg, pos = 0,0

    for i in index_vec[1:]:
      if trainlabels[i] == label1: 
          pos += 1
      else: 
          neg += 1
	label2 = trainlabels[i]
    if pos >= neg:
        return label1
    else:
        return label2


def calculate(traindata, trainlabels, k):
    distances = calculate_distances(trainlabels, traindata)
    correct = 0.0
    total = len(traindata)
    for x in range(len(traindata)):
	    index_vec = search(x, distances[x], k)
	    classification = classify(index_vec, trainlabels)
            if trainlabels[x] == classification[0]:
		correct += 1
    return correct/total


if __name__ == '__main__':
    
    k = sys.argv[1] if len(sys.argv) == 2 else 5
    
    if len(sys.argv) == 3:
        datafile = sys.argv[2]
    else:
        datafile= "dataset"

    traindata, trainlabels, testdata, evaldata, testlabels = load_data(datafile)
    print(calculate(traindata, trainlabels, k))
    

