#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

#reading file
my_file = pd.read_csv('Madaline.csv', header = None)
X = my_file[0]
Y = my_file[1]
x1 = []
y1 = []
x2 = []
y2 = []
target_ = my_file[2]
for i in range(len(target_)):
    if target_[i] == 0:
        target_[i] = -1
        x1.append(X[i])
        y1.append(Y[i])
    elif target_[i] == 1:
        x2.append(X[i])
        y2.append(Y[i])


# In[2]:


def plot_datas(x1,y1,x2,y2):
    plt.scatter(x1, y1, label='class1')
    plt.scatter(x2, y2, label='class2')
    plt.xlabel('X 1')
    plt.ylabel('X 2')
    plt.legend()
    plt.show()
    
plot_datas(x1,y1,x2,y2)


# In[3]:


def plot_classes_with_lines(w,b,num_of_nodes):
    plt.scatter(x1, y1, label='class1')
    plt.scatter(x2, y2, label='class2')
    plt.xlabel('X 1')
    plt.ylabel('X 2')
    
    x = np.linspace(-2,2,100)
    for i in range(num_of_nodes):
        y = (-w[i]*x -b[i])/w[i+num_of_nodes]
        plt.plot(x,y, label = str(i+1))
    plt.legend()
    plt.ylim([-3, 3])
    plt.show()


# In[4]:


#Madaline - MRI
def MRI(num_of_nodes, learning_rate_, epochs_):
    learning_rate = learning_rate_
    random.seed(1)
    w = []
    b = []
    #initializing weights
    for i in range(2*num_of_nodes):
        w.append(random.random())
    for i in range(num_of_nodes):
        b.append(random.random())
        
    #shuffling and my final variables will be X1, X2 & target
    X1_ = np.concatenate([x1, x2])
    X2_ = np.concatenate([y1, y2])
    shuffler = np.random.permutation(len(target_))
    X1 = X1_[shuffler]
    X2 = X2_[shuffler]
    target = target_[shuffler]
    
    def mysign(x):
        if x >= 0: 
            return 1
        else:
            return -1
        
    #step 4 OK
    def calculate_net(i, node_num):
        return w[node_num-1]*X1[i] + w[node_num-1+num_of_nodes]*X2[i] + b[node_num-1]

    #step 5
    def hidden_layer_output(i, node_num):
            return mysign(calculate_net(i, node_num))

    #step 6 (or)
    def net_output(i):
        for j in range(num_of_nodes):
            if hidden_layer_output(i, j+1) == 1:
                return 1
        return -1

    #step 7
    epochs = 0
    while epochs != epochs_:
        
        #shuffling datas on every epoch
        X1_ = np.concatenate([x1, x2])
        X2_ = np.concatenate([y1, y2])
        shuffler = np.random.permutation(len(target_))
        X1 = X1_[shuffler]
        X2 = X2_[shuffler]
        target = target_[shuffler]
        
        epochs += 1
        for i in range(len(target)):
            if target[i] == net_output(i):
                continue

            if target[i] == 1:
                Z_f = 0
                Z_f_net = 10**9 #large number
                for j in range(num_of_nodes):
                    j =j+1
                    if np.abs(calculate_net(i, j)) < Z_f_net:
                        Z_f_net = np.abs(calculate_net(i, j))
                        Z_f = j
                net = calculate_net(i, Z_f)
                b[Z_f-1] += learning_rate*(1 - net)
                w[Z_f-1] += learning_rate*(1 - net)*X1[i]
                w[Z_f-1+num_of_nodes] += learning_rate*(1 - net)*X2[i]
                
            if target[i] == -1:
                for j in range(num_of_nodes):
                    j += 1
                    if np.abs(calculate_net(i, j)) > 0:
                        
                        net = calculate_net(i, j)
                        b[j-1] += learning_rate*(-1 - net)
                        w[j-1] += learning_rate*(-1 - net)*X1[i]
                        w[j-1+num_of_nodes] += learning_rate*(-1 - net)*X2[i]
    return w, b


# ### with 3 nodes! not working good!

# In[5]:


#working in learning rate of 0.0001 and epochs of 150
num_of_nodes = 3
learning_rate = 0.0001
epochs = 150
w, b = MRI(num_of_nodes, learning_rate, epochs)
plot_classes_with_lines(w, b, num_of_nodes)
for i in range(num_of_nodes):
    print('line',str(i+1),':')
    print('w_x:', w[i],'w_y:', w[i+num_of_nodes])
    print('bias:', b[i])


# ### with 4 nodes, kind of working!

# In[6]:


#working in learning rate of 0.00004 and epochs of 120
num_of_nodes = 4
learning_rate = 0.00004
epochs = 120
w, b = MRI(num_of_nodes, learning_rate, epochs)
plot_classes_with_lines(w, b, num_of_nodes)
for i in range(num_of_nodes):
    print('line',str(i+1),':')
    print('w_x:', w[i],'w_y:', w[i+num_of_nodes])
    print('bias:', b[i])


# ### with 8 nodes! just 3 lines working!

# In[7]:


#working in learning rate of 0.00004 and epochs of 450
num_of_nodes = 8
learning_rate = 0.00004
epochs = 450
w, b = MRI(num_of_nodes, learning_rate, epochs)
plot_classes_with_lines(w, b, num_of_nodes)
for i in range(num_of_nodes):
    print('line',str(i+1),':')
    print('w_x:', w[i],'w_y:', w[i+num_of_nodes])
    print('bias:', b[i])


# ## Testing code

# ### test1: XOR

# In[14]:


#XOR
x1 = np.array([1,-1])
y1 = np.array([1,-1])
x2 = np.array([1,-1])
y2 = np.array([-1,1])
target_ = np.array([-1,-1,1,1])
plot_datas(x1,y1,x2,y2)


# In[19]:


num_of_nodes = 2
learning_rate = 0.5
epochs = 50
w, b = MRI(num_of_nodes, learning_rate, epochs)
plot_classes_with_lines(w, b, num_of_nodes)
for i in range(num_of_nodes):
    print('line',str(i+1),':')
    print('w_x:', w[i],'w_y:', w[i+num_of_nodes])
    print('bias:', b[i])


# ### test2: my data

# In[20]:


#my data
x1 = np.array([1,1,1,-1,-1,-1])
y1 = np.array([-0.5,0,0.5,-0.5,0,0.5])
x2 = np.array([0,0,0])
y2 = np.array([-0.5,0,0.5])
target_ = np.array([1,1,1,1,1,1,-1,-1,-1])
plot_datas(x1,y1,x2,y2)


# In[22]:


num_of_nodes = 2
learning_rate = 0.8
epochs = 50
w, b = MRI(num_of_nodes, learning_rate, epochs)
plot_classes_with_lines(w, b, num_of_nodes)
for i in range(num_of_nodes):
    print('line',str(i+1),':')
    print('w_x:', w[i],'w_y:', w[i+num_of_nodes])
    print('bias:', b[i])

