# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 21:05:05 2017

@author: longmaisg
"""

"""
There are 6 squares. Reward = 100 when arriving square 5
Square:
[0|0|0]
[0|0|1]
"""

import numpy as np
import random

        
# Generate Q matrix
numberOfState = 6
Q = np.zeros((numberOfState,numberOfState))

# Reward = 100 if current state is reward state
rewardState = 5
reward = 100
gamma = 0.5
alpha = 0.1
nextState = 0

for explore in range (1000):
    
    currentState = random.choice(np.arange(rewardState))
    #currentState = 0
    
    #while (currentState != rewardState):
    #for i in range (20):
    while True:
        if (currentState == 0):
            foo = [1,3]
        elif (currentState == 1):
            foo = [0,2,4]
        elif (currentState == 2):
            foo = [1,5]
        elif (currentState == 3):
            foo = [0,4]
        elif (currentState == 4):
            foo = [1,5]
        
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
    


    

