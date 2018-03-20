# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 21:05:05 2017

@author: longmai
"""

import numpy as np
import random

        
# Generate Q matrix
numberOfState = 16
Q = np.zeros((numberOfState,numberOfState))

# Reward = 100 if current state is reward state
rewardState = 15
reward = 1000
gamma = 0.5
alpha = 0.1
nextState = 0
removeState = [2,9,10,14]

for explore in range (1000):
    
    availState = np.delete(np.arange(rewardState), removeState)
    currentState = random.choice(availState)
    #currentState = random.choice(np.arange(rewardState))
    #currentState = 0
    
    #while (currentState != rewardState):
    #for i in range (20):
    while True:
        if (currentState == 0):
            foo = [1,4]
        elif (currentState == 1):
            foo = [0,5]
        elif (currentState == 3):
            foo = [7]
        elif (currentState == 4):
            foo = [0,5,8]
        elif (currentState == 5):
            foo = [1,4,6]
        elif (currentState == 6):
            foo = [5,7]
        elif (currentState == 7):
            foo = [3,6,11]
        elif (currentState == 8):
            foo = [4,12]
        elif (currentState == 11):
            foo = [7,15]
        elif (currentState == 12):
            foo = [8,13]
        elif (currentState == 13):
            foo = [12]
        
        """
        a = max(Q[currentState,:])
        for j in foo:
            if (Q[currentState, j] == a):
                nextState = j
        #print(currentState, j, '\n')
            #else:
                #nextState = random.choice(foo)  
                """
        nextState = random.choice(foo) 
            
        r = 0
        if (nextState == rewardState):
            r = reward    
        Q[currentState, nextState] = r + gamma*max(Q[nextState,:])
        Q[currentState, nextState] = (1 - alpha)*Q[currentState, nextState] + alpha*Q[currentState, nextState]
        currentState = nextState
        if (currentState == rewardState):
            break
        
print(Q)
    


    

