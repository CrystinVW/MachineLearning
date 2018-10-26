from __future__ import division
import numpy as np
from scipy.sparse import csc_matrix
np.set_printoptions(threshold='nan')


def naive_bayes(features, labels):
    ''' compute the naive bayes learner '''
    # count number each label for prior
    neginds = (labels == 0.0).nonzero()[0]
    posinds = (labels == 1.0).nonzero()[0]

    a1 = len(neginds)/len(labels)
    a2 = len(posinds)/len(labels)
    label_counts = np.array([a1, a2]).flatten()
    label_prob = np.log(label_counts)
    
    # dividing the feature data into pos and neg 
    aux_neg = features[neginds, :]
    aux_pos = features[posinds, :]
    
    # summing over for each document (first field)
    # adding smoothing, log to prevent overflow
    param_neg = np.log((1 + aux_neg.sum(0)) / (1+aux_neg.sum(0)).sum())
    param_pos = np.log((1 + aux_pos.sum(0)) / (1+aux_pos.sum(0)).sum())
    pn = np.squeeze(np.asarray(param_neg))
    pp = np.squeeze(np.asarray(param_pos))
    params = [pn, pp]
    
    return label_prob, params

     
def classify_naive_bayes(params, label_probs, features):
    ''' compute the naive bayes classifier '''
    #create an array with zeros in the size of test (200)
    laux = np.zeros((np.size(features, axis=0), 1))
    labels = np.squeeze(np.asarray(laux)) 

    # find conditional proba. for each class
    # find most likely label for each instance
    for i in range(np.size(features, axis=0)):
    	cp1 = features[i,:]*params[0]+label_probs[0]
  	cp2 = features[i,:]*params[1]+label_probs[1]
        if cp1 > cp2:
        	j = 0
	else:
		j = 1
 
	labels[i]=j
    return labels

def train_logistic_regression(features, labels, l_rate, target_delta, reg_constant):
    ''' gradient ascent algorithm to train logistic regression params,
        where offset and weights are the parameters '''
    offset = 0
    w_delta = 100000

    # old_ws and ws are matrices 1x 10770 with zeros
    old_ws = np.zeros((1, np.size(features, axis=1)-1))
    ws =  np.zeros((1, np.size(features, axis=1)))
    
    # regularizer is a matrix 1x10770 multiplied by reg_constant
    regularizer = np.zeros((1, np.size(features, axis=1)))*reg_constant
  
    ''' calculate the probability that each instance is classified as 1 '''
    ''' first calculate the weight sums '''
    # create a 200x1 array with the value of offset
    # same as repmat(offset, size(features,1),1) in MATLAB
    w_1 = np.ones((np.size(features, axis=0),1))* offset

    # create a 200x10770 aarray with the value of ws
    # same as repmat(ws, size(features,1),1) in MATLAB    
    w_2 =   np.ones((np.size(features, axis=0), 1)) * ws

    # multiply these by features (which is 200x10770)
    w_3 =  (np.multiply(w_2, features.todense())).sum(1)
    
    # finally calculates the weight sums 200x1
    w_sums = w_1 + w_3
    
    ''' now comput the probabilities '''
    # creating logistic
    den = np.exp(w_sums) + 1
    num = np.exp(w_sums)
    probs = num/den
    
    ''' calculating current ll '''
    # expand 1 dim in labels to 200x1
    l_2d = np.expand_dims(labels,1)
    c_aux = np.multiply(w_sums[:np.size(w_sums, axis=0)-1], l_2d)
    c_aux2 = np.log(1 + np.exp(w_sums[:np.size(w_sums, axis=0)-1]))
    c_aux3 = (c_aux - c_aux2).sum()
    c_aux4 = np.multiply(np.multiply(ws,ws), (regularizer//2))
    c_aux5 = c_aux4.sum()
    current_ll = c_aux3 - c_aux5

    print("Training logistic regression. Initial: ", current_ll)
	
    '''starting iterations '''
    iter_n = 0
    probss = probs[:np.size(probs)-1]
    featuress =  features [:np.size(features, axis=0)-1,:np.size(features, axis=1)-1]
    wss= ws[:, :np.size(ws, axis=1)-1]
    regularizers = regularizer[:,:np.size(regularizer, axis=1)-1]

    while (w_delta > target_delta):
	old_ws[:] = wss[:]

        # calculating the gradient 
        grad_aux0 = (l_2d  - probss)
        grad_aux00 = np.ones((1, np.size(features, axis=1)-1))
        grad_aux = grad_aux0*grad_aux00
        grad_aux1 = np.multiply(grad_aux, featuress.todense())
        grad_aux2 = csc_matrix(grad_aux1.sum(0))
        grad_aux3 = np.multiply(regularizers, wss)
        grad_aux4 = grad_aux3[:, :np.size(grad_aux3, axis=1)-1]
        grad = grad_aux2 - grad_aux3	

	# magnitude limit gradient
        grad = l_rate*grad
        iter_n += 1

        # update ws with previous labe prob
        offset = offset +  l_rate*grad_aux0.sum()
        wss = wss + grad

	# using the current weights, calculate the proba for instance 
    	w_1 = np.ones((np.size(featuress, axis=0),1))* offset  
   	w_2 =   np.ones((np.size(featuress, axis=0), 1)) * wss
    	w_3 =  (np.multiply(w_2, featuress.todense())).sum(1)
   	w_sums = w_1 + w_3

        # iterating logist
	den = np.exp(w_sums) + 1
    	num = np.exp(w_sums)
    	probss = num/den

	# update likelihood
    	l_2d = np.expand_dims(labels,1)
    	c_aux = np.multiply(w_sums, l_2d)
    	c_aux2 = np.log(1 + np.exp(w_sums))
    	c_aux3 = (c_aux - c_aux2).sum()
    	c_aux4 = np.multiply(np.multiply(wss,wss), (regularizers//2))
    	c_aux5 = c_aux4.sum()
    	current_ll = c_aux3 - c_aux5
	
	# update weight delta
	w_delta = np.sqrt((  np.multiply((old_ws - wss),(old_ws - wss)) ).sum() )

	# print?
  	if (np.mod(iter_n, 100) == 0):
		print('Log-likelihood, weight delta: ', current_ll, w_delta)

    print('Final ll:', current_ll)
    return  offset, wss 



def run_logistic_regression(offset, wss, features):
	featuress =  features [:np.size(features, axis=0)-1,:np.size(features, axis=1)-1]
    	# using the current weights, calculate the proba for instance 
    	w_1 = np.ones((np.size(featuress, axis=0),1))* offset  
   	w_2 =   np.ones((np.size(featuress, axis=0), 1)) * wss
    	w_3 =  (np.multiply(w_2, featuress.todense())).sum(1)
   	w_sums = w_1 + w_3

    	posinds = (w_sums > 0).nonzero()[0]

	laux = np.zeros((np.size(features, axis=0), 1))
    	labels = np.squeeze(np.asarray(laux)) 

	labels[posinds] = 1
    	return labels 

def load_data(filename):
    ''' load the data and the label files for
        either training or test data '''   
    c1, c2, c3 = np.loadtxt(filename + ".data", unpack=True)
    # sames as training_data = spconvert(loaded_file); in MATLAB
    counts = csc_matrix((c3, (c1, c2)))
    labels = np.loadtxt(filename + ".label", unpack=True)
    return counts, labels



def print_results(string, result_train, train_labels, result_test, test_labels):
    ''' return the accuracy for test and training '''
    accuracy_train = len(result_train)/len(train_labels)
    print(string +  'Train accuracy: ' , accuracy_train)
    
    accuracy_test = len(result_test)/len(test_labels)
    print(string +  'Test accuracy: ' , accuracy_test)




def main():
    #load training and test data
    train, train_labels = load_data("train")
    test, test_labels = load_data("test")

    # run naive bayes
    nb_label_probs, nb_params = naive_bayes(train, train_labels)
    nb_train_labels = classify_naive_bayes(nb_params, nb_label_probs, train)
    nb_labels = classify_naive_bayes(nb_params, nb_label_probs, test)
 
    # print results
    result_test_nb = []
    for i in range(len(test_labels)):
	if nb_labels[i] == test_labels[i]:
		result_test_nb.append(i)
  
    result_train_nb = []
    for i in range(len(train_labels)):
	if nb_train_labels[i] == train_labels[i]:
		result_train_nb.append(i)
    print_results('Naive Bayes ',result_train_nb,train_labels,result_test_nb, test_labels)
    

    # run logistic regression
    l_rate = 0.0001
    target_delta = 0.001
    reg_constant = 0
    lr_offset, lr_w = train_logistic_regression(train, train_labels,l_rate, target_delta, reg_constant)   
    
    lr_train_labels = run_logistic_regression(lr_offset, lr_w, train)
    lr_labels = run_logistic_regression(lr_offset, lr_w, test)

    
    # print results
    result_test_lr = []
    for i in range(len(test_labels)):
	if lr_labels[i] == test_labels[i]:
		result_test_lr.append(i)
  
    result_train_lr = []
    for i in range(len(train_labels)):
	if lr_train_labels[i] == train_labels[i]:
		result_train_lr.append(i)
    print_results('Logistic Regression ',  result_train_lr,  train_labels, result_test_lr, test_labels)

   
    print("Done!")



if __name__ == '__main__':
    main()
