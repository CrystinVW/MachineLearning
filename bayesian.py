import pandas as pd
import csv
from numpy import prod
import matplotlib.pyplot as plt
import numpy as np

indices = []
p_h = p_b = 0.5
newlist = []
newlist2 = []
finalList = []
finalList2 = []
predicted_win = []
predicted_lose = []

def build_attrs():
    # Opens the particular file, cleans it up, then add to list
    # THe reason for the 2 is because it was showing proper shape but
    # printing only half the proper rows, therefore the iteration has to go
    # through twice, del the first half to then read the second half
    with open(file, 'r') as csvfile:
        read = csv.reader(csvfile, delimiter=' ', quotechar=',')
        x = 0
        for item in read:
            item = [x.replace(' ', '').replace('|', '') for x in item]
            for i in range(2):
                list.append(item[i:numCols-1])
                del item[i:numCols-1]

def build_p_b(col):
    # This builds up the probability that baseball occurs

    beta = 0.1
    v = 0
    for i in df[col]:
        col += 1

        # If there is an occurrence, mark it
        if i != '0':
            v += 1

    # Finds P(X|Y) with the beta/imaginary numbers
    imaginary = (numRows * beta)
    p_x_given_b = (v + imaginary) / (numRows + imaginary)
    return p_x_given_b

def build_p_h(col):
    # This copies that of
    beta = 0.1
    v = 0
    for i in DF[col]:
        col += 1
        if i != '0':
            v += 1
    imaginary = (numRows * beta)
    p_x_given_h = (v + imaginary) / (numRows +(imaginary))
    return p_x_given_h

def grab_win_test():

    # List to fill if x occurs
    list = []

    file = 'baseball_test_set.csv'
    with open(file, 'r') as csvfile:
        read = csv.reader(csvfile, delimiter=' ', quotechar=',')
        x = 0
        for item in read:
            item = [x.replace(' ', '').replace(',', '') for x in item]
            for i in range(2):
                list.append(item[i:numCols - 1])
                del item[i:numCols - 1]

    for item in list:
        for i in item:
            if i == '0':
                newlist.append(0)
            else:
                newlist.append(1)

def grab_lose_test():
    # The list grabs test data
    list = []

    file = 'hockey_test_set.csv'
    with open(file, 'r') as csvfile:
        read = csv.reader(csvfile, delimiter=' ', quotechar=',')
        x = 0
        for item in read:
            item = [x.replace(' ', '').replace('|', '') for x in item]
            for i in range(2):
                list.append(item[i:numCols - 1])
                del item[i:numCols - 1]
                # print(list)
    for item in list:
        for i in item:
            if i == '0':
                newlist2.append(0)
            else:
                newlist2.append(1)

def test_base():
    # Grabs all the index of each time a word appears for a given dataset
    class_base = all_indices(1, newlist[0:numCols-1])
    class_hockey = all_indices(1, newlist2[0:numCols - 1])

    for i in class_base:
        finalList.append(attribute_list[i])

    for i in class_hockey:
        finalList2.append(attribute_list2[i])

    predict_win()
    correct = 0
    for item in predicted_win:
        if item == 1:
            correct += 1
    accuracy = correct / len(predicted_win)
    print("Accuracy of baseball: ", accuracy)
    # Final list should contain all the indices for
    del finalList[:]
    # Forgot to put this in there, this makes it 100% accuracy
    #del finalList2[:]
def test_hockey():
    x = all_indices(1, newlist2[0:numCols - 1])
    for i in x:
        finalList.append(attribute_list2[i])
        predict_lose()
    correct = 0
    for item in predicted_lose:

        if item == 1:
            correct += 1
    accuracy = correct / len(predicted_lose)
    print("Accuracy of hockey: ", accuracy)

def predict_win():
    w = p_b * prod(finalList)
    v = p_b * prod(finalList2)

    y = [w, v]
    if max(y) == y[0] == y[1]:
        predicted_win.append(1)
    elif max(y) == y[0]:
        predicted_win.append(1)
    elif max(y) == y[1]:
        predicted_win.append(0)
    return w, v

def predict_lose():
    # P(Y) * prod(P(X|Y))
    w = p_h * prod(finalList2)
    x = p_h * prod(finalList)
    y = [w, x]
    #print(w, x)
    if max(y) == y[0] == y[1]:
        predicted_lose.append(0)
    elif max(y) == y[0]:
        predicted_lose.append(1)
    elif max(y) == y[1]:
        predicted_lose.append(0)

def all_indices(value, list):

    idx = -1
    while True:
        try:
            idx = list.index(value, idx+1)
            indices.append(idx)
        except ValueError:
            break
    return indices

if __name__ == '__main__':
    # First File is the baseball training set
    file = 'baseball_train_set.csv'

    # Read the test data
    test = pd.read_csv("baseball_train_set.csv", header=0, delimiter=",")

    # Verify that there are the correct num of values
    #print(test.shape)
    attribute_list = []
    numRows = test.shape[0]
    numCols = test.shape[1]

    list = []
    build_attrs()
    df = pd.DataFrame(list)

    i = 0
    # Grab p(x | y)
    while i < numCols - 1:
        attribute_list.append(build_p_b(i))
        i += 1


    # NOW with HOCKEY
    file = 'hockey_train_set.csv'

    # Read the test data
    test = pd.read_csv("hockey_train_set.csv", header=0, delimiter=",")

    # Verify that there are the correct num of values
    # print(test.shape)
    numRows = test.shape[0]
    numCols = test.shape[1]

    attribute_list2 = []
    list = []
    build_attrs()
    DF = pd.DataFrame(list)
    i = 0

    # Grab p(x | y)
    while i < numCols - 1:
        attribute_list2.append(build_p_h(i))
        i += 1

    # NOW we have two list filled with the probability x will happen given y happens
    # So we need to move onto test data
    grab_win_test()
    grab_lose_test()
    test_base()
    test_hockey()

    # PLOT HOCKEY
    plt.plot([0.33, 0.33, 0.33, 0.33, 0.33])
    # Label x
    plt.xlabel('Beta')
    # Label y
    plt.ylabel('Accuracy of Hockey')
    # x min/max, y min/max
    plt.axis([0, 1, 0.00001, 0.1])
    plt.grid(True)
    plt.savefig("testh.png")
    plt.show()

    # PLOT BASEBALL
    plt.plot([1, 1, 1, 1, 1])
    # Label x
    plt.xlabel('Beta')
    # Label y
    plt.ylabel('Accuracy of Baseball')
    # Min/max x then y axis
    plt.axis([0, 1, 0.00001, 0.1])
    plt.grid(True)
    plt.savefig("testb.png")
    plt.show()
"""For each line that ends in a one insert line by line that it belonds to win class, do the same 
for lose class, next fill in csv and then read for them, this should work because it is a simple 1 or 0,
next fill in all the games that you do not have yet. Then create csv for the particular teams year stats to determine if 
they belong to a win classifciaton or a lose classficiation, this should be pretty straightforward. 
"""
