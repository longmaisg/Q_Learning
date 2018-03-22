# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:54:23 2017

@author: longmai
"""

# Topology 1

"""
This algorithm works with network of more than 20 nodes and fewer 3 sinks
"""

import numpy as np
import random
import time
from copy import deepcopy

def signal_handler(signal, frame):
    global interrupted
    interrupted = True
    
def feedforward(a0,W1,b1,W2,b2,W3,b3,returnAll = 1):    
    """ Layer 1 """
    a1 = np.dot(W1,a0) + b1
    
    """Layer 2 """
    a2 = np.dot(W2,a1) + b2
    
    """ Layer 3"""
    a3 = float(np.dot(W3,a2) + b3)
    if (returnAll == 1):
        return a1,a2,a3
    else:
        return a3
    
# Count processing time
startTime = time.time()
      
"""
--------- Insert input state --------------
Input data include: 
    - State transition matrix
    - Start state (assume that there is only 1 start state)
    - Sink states (all sink states is listed in a list)
    - Reward states represent sink states. Reward states format is similar to sink states
    - Gamma parameter, for Q-learning in Reinforcement Learning
    - Alpha parameter, for Q-learning in Reinforcement Learning 
"""

sinkState = [3,6]
terminal = np.array((3,6))
startState = 0
reward = 100
gamma = 0.5
alpha = 0.1
maxD = 400

#stateTransitionMatrix
"""
------------ Generate Transition Matrix for each state ---------------
Assume that rows represent current states. Columns represent next state. 
Hence, the size of this matrix is [number of states, number of states]
"""

T = np.array(                    [  [1,1,0,0,1,0,0,0],
                                    [1,1,1,0,0,0,0,0],    
                                    [0,1,1,1,0,0,1,0],
                                    [0,0,1,1,0,1,0,1],
                                    [1,0,0,0,1,1,0,0],
                                    [0,0,0,1,1,1,0,0],
                                    [0,0,1,0,0,0,1,1],
                                    [0,0,0,1,0,0,1,1]], dtype=np.int)

# number of total states in this topology
numberOfNodes = len(T[0])
nodes = len(T)
numberOfAction = pow(nodes, len(sinkState))

sinks = np.zeros(len(terminal), dtype=np.int)
"""e is tuple to store current state, next state (action), and reward """
D = []
e = [0,0,0,0] #[s,a=s-s',r,s']
startState = np.zeros((nodes,1), dtype=np.int)
#sinks = np.zeros(len(terminal), dtype=np.int)

for generateDataset in range(10):
    interrupted = False
    sinks = np.array((random.choice(np.arange(numberOfNodes)), random.choice(np.arange(numberOfNodes))))
    #print (deepcopy(sinks))
    while True:
        """ 
        ---------------- Choose random action ---------------
        From current state, for example [2,3], choose random action for all sinks. 
        This step reduce the number of actions that
        we need to choose
        """
        state = np.zeros((nodes,1), dtype=np.int)
        for k in sinks:
            state[k] += 1
        e[0] = deepcopy(state.astype(np.float32))
        #temp = 0
        for x in range(len(sinks)):
            foo = np.transpose(np.where(T[sinks[x],:] != 0))
            foo = int(random.choice(foo)) 
            sinks[x] = foo
            #temp += sinks[x]*pow(numberOfNodes,len(sinks)-x-1)
        state1 = np.zeros((nodes,1), dtype=np.int)
        for k in sinks:
            state1[k] += 1
        e[1] = deepcopy((state1-state).astype(np.float32))
        e[3] = deepcopy(state1)
        e[2] = 0
        #print (deepcopy(sinks))
        if ((sinks==terminal).all()):
            e[2] = 100
            D.append(deepcopy(e))
            break
        D.append(deepcopy(e))
        if (len(D) > maxD):
            del D[0:200]
        if interrupted:
            print("Gotta go")
            break
    #print ("finish loop", generateDataset, "to generate dataset")
print ("Finish generating D")
      
eta = 0.001
nodeOfLayer1 = 20
nodeOfLayer2 = 20
nodeOfLayer3 = 1
W1 = 0.01*np.random.randn(nodeOfLayer1,numberOfNodes)
b1 = 1 + 0.01*np.random.randn(nodeOfLayer1,1)
W2 = 0.01*np.random.randn(nodeOfLayer2,nodeOfLayer1)
b2 = 1 + 0.01*np.random.randn(nodeOfLayer2,1)
W3 = 0.01*np.random.randn(nodeOfLayer3,nodeOfLayer2)
b3 = 1 + 0.01*np.random.randn(nodeOfLayer3,1)
"""
W3 = 0.01*np.random.randn(numberOfAction,nodeOfLayer2)
b3 = 1 + 0.01*np.random.randn(numberOfAction,1)"""
minibatch = 10
epsilon = 0.01
alpha = 0.1

"""
Do forever loop:
Generate dataset
Use dataset to train for action probability network
Use action probability network to generate new dataset:
    - Choose the best action with epsilon-greedy
"""
for iteration in range(10):
    """ Training for action probability network """
    W3temp = 0
    b3temp = 0
    W2temp = 0
    b2temp = 0
    W1temp = 0
    b1temp = 0
    for epoch in range (10):
        for batch in range(minibatch):
            """ Feed Forward """
            
            """ Layer 1 """
            a0 = random.choice(D)
            a1,a2,a3 = feedforward(a0[1],W1,b1,W2,b2,W3,b3)
            
            """ Find Q-max of next state"""
            Qmax = 1e-5
            for findQmax in range(len(D)):
                if (D[findQmax][0].all() == a0[3].all()):
                    QmaxTemp = feedforward(a0[1],W1,b1,W2,b2,W3,b3,0)
                    if (QmaxTemp > Qmax):
                        Qmax = QmaxTemp
                    
            """ Find target"""
            y = float((1-alpha)*a3 + alpha*(a0[2] + Qmax))
            #print ("y, a3:", y, a3)
                      
            """ Back Propagation """
            lost = abs(y - a3)
            delta3 = lost*1
            delta2 = np.dot(np.transpose(W3),delta3)*1
            delta1 = np.dot(np.transpose(W2),delta2)*1
                           
            W3temp += np.dot(delta3,np.transpose(a2))
            b3temp += delta3
            W2temp += np.dot(delta2,np.transpose(a1))
            b2temp += delta2
            W1temp += np.dot(delta1,np.transpose(a0[1]))
            b1temp += delta1
        
        W3 -= (eta/minibatch)*W3temp
        b3 -= (eta/minibatch)*b3temp
        W2 -= (eta/minibatch)*W2temp
        b2 -= (eta/minibatch)*b2temp
        W1 -= (eta/minibatch)*W1temp
        b1 -= (eta/minibatch)*b1temp
                       
        #print ("finish training with epoch", epoch)
    #print ("finish training with iteration", iteration)
    
    """ Generate new dataset with action probability network """    
    D = []
    e = [0,0,0,0] #[s,a=s-s',r,s']
    #sinks = np.zeros(len(terminal), dtype=np.int)
    for generateDataset in range(1):
        interrupted = False
        sinks = np.array((random.choice(np.arange(numberOfNodes)), random.choice(np.arange(numberOfNodes))))
        #print (sinks)
        test = []
        test.append(deepcopy(sinks))
        while True:
            """ 
            ---------------- Choose random action ---------------
            From current state, for example [2,3], choose random action for all sinks. 
            This step reduce the number of actions that
            we need to choose
            """
            state = np.zeros((nodes,1), dtype=np.int)
            for k in sinks:
                state[k] += 1
            e[0] = deepcopy(state.astype(np.float32))
            temp = 0
            """ Choose the best action with epsilon-greedy """
            """ With probability of epsilon, choose random action"""
            if (random.random() < epsilon):
                for x in range(len(sinks)):
                    foo = np.transpose(np.where(T[sinks[x],:] != 0))
                    foo = int(random.choice(foo)) 
                    sinks[x] = foo
                
            else:
                """ Choose action that lead to max Q value in next state"""
                Qmax = 1e-5
                foo = []
                for x in range(len(sinks)):
                    """ Choose all possible action"""
                    foo.append(np.where(T[sinks[x],:] != 0))
                    
                sinksTemp = np.zeros(len(terminal), dtype=np.int)
                """
                Using feed forward neural network to find Q value of all possible actions
                sinksTemp store temporary sinks value
                (sinks is the position of each sink, that is where each sink in the network).
                From sinksTemp, count next state for each action. Using next state to count
                Q value and choose action with max Q value
                """
                for l0 in range(len(foo[0][0])):
                    sinksTemp[0] = foo[0][0][l0]
                    for l1 in range(len(foo[1][0])):
                        sinksTemp[1] = foo[1][0][l1]
                        #print (sinksTemp)
                        state1 = np.zeros((nodes,1), dtype=np.int)
                        for k in sinksTemp:
                            state1[k] += 1
                        QmaxTemp = feedforward(state1-state,W1,b1,W2,b2,W3,b3,0)
                        #print (QmaxTemp)
                        if (QmaxTemp > Qmax):
                            sinks = sinksTemp
            test.append(deepcopy(sinks))
            state1 = np.zeros((nodes,1), dtype=np.int)                
            for k in sinks:
                state1[k] += 1
            e[1] = deepcopy((state1-state).astype(np.float32))
            e[3] = deepcopy(state1)
            e[2] = 0
            if ((sinks==terminal).all()):
                e[2] = 100
                D.append(deepcopy(e))
                break
            D.append(deepcopy(e))  
            if (len(D) > maxD):
                del D[0:200]
            if interrupted:
                print("Gotta go")
                break
print ("Finish training")     

""" Test script """
sinks = np.array((2,2))
Qmax = 1e-5
foo = []
for x in range(len(sinks)):
    """ Choose all possible action"""
    foo.append(np.where(T[sinks[x],:] != 0))
    
sinksTemp = np.zeros(len(terminal), dtype=np.int)
for l0 in range(len(foo[0][0])):
    sinksTemp[0] = foo[0][0][l0]
    for l1 in range(len(foo[1][0])):
        sinksTemp[1] = foo[1][0][l1]
        #print (sinksTemp)
        state1 = np.zeros((nodes,1), dtype=np.int)
        for k in sinks:
            state1[k] += 1
        QmaxTemp = feedforward(state1-state,W1,b1,W2,b2,W3,b3,0)
        print (sinksTemp, QmaxTemp)
        if (QmaxTemp > Qmax):
            Qmax = QmaxTemp
            sinks = sinksTemp
print (sinks)

# Count processing time    
stopTime = time.time()
print ("\nTimer: ", stopTime - startTime)