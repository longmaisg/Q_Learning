# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:42:18 2017

@author: longmai
"""

import numpy as np
import random

        
# Generate Q matrix
numberOfState = 9
Q = np.zeros((numberOfState,numberOfState))

# Reward = 100 if current state is reward state
endState = 8
rewardState1 = 8
rewardState2 = 7
reward1 = 100
reward2 = -100
gamma = 0.5
alpha = 0.1
nextState = 0
removeState = [endState]

for explore in range (1000):
    
    availState = np.delete(np.arange(endState), removeState)
    currentState = random.choice(availState)
    #currentState = random.choice(np.arange(rewardState))
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
            foo = [0,4,6]
        elif (currentState == 4):
            foo = [1,3,5,7]
        elif (currentState == 5):
            foo = [2,4,8]
        elif (currentState == 6):
            foo = [3,7]
        elif (currentState == 7):
            foo = [4,6,8]
        elif (currentState == 8):
            foo = [5,7]
        
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
        if (nextState == rewardState1):
            r = reward1
        elif (nextState == rewardState2):
            r = reward2
        Q[currentState, nextState] = r + gamma*max(Q[nextState,:])
        Q[currentState, nextState] = (1 - alpha)*Q[currentState, nextState] + alpha*Q[currentState, nextState]
        currentState = nextState
        if (currentState == endState):
            break
        
print(Q)
    


    

