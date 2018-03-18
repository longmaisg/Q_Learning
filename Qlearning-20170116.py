# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 13:50:29 2017

@author: longmai
"""


"""
There are 9 squares. Reward = 100 when arriving square 8
"""

import numpy as np
import random

        
# Generate Q matrix
numberOfState = 9
Q = np.zeros((numberOfState,numberOfState))

# Reward = 100 if current state is reward state
endState = 8
rewardState1 = 8
rewardState2 = 8
reward1 = 100
reward2 = 100
gamma = 0.5
alpha = 0.1
nextState = 0
removeState = [endState]

stateTransitionMatrix = np.array([  [0,1,0,1,0,0,0,0,0],
                                    [1,0,1,0,1,0,0,0,0],    
                                    [0,1,0,0,0,1,0,0,0],
                                    [1,0,0,0,1,0,1,0,0],
                                    [0,1,0,1,0,1,0,1,0],
                                    [0,0,1,0,1,0,0,0,1],
                                    [0,0,0,1,0,0,0,1,0],
                                    [0,0,0,0,1,0,1,0,1],
                                    [0,0,0,0,0,1,0,1,0]])

for explore in range (100):
    
    availState = np.delete(np.arange(numberOfState), removeState)
    currentState = random.choice(availState)

    while True:
                
        foo = np.where(stateTransitionMatrix[currentState,:] != 0)
        foo = np.transpose(foo)
        nextState = int(random.choice(foo) )
        #nextState = random.choice(np.arange(numberOfState))
            
        r = 0
        if (nextState == rewardState1):
            r = reward1
        elif (nextState == rewardState2):
            r = reward2
        temp = stateTransitionMatrix[currentState,nextState]*(r + gamma*max(Q[nextState,:]))
        Q[currentState, nextState] = (1 - alpha)*Q[currentState, nextState] + alpha*temp
        currentState = nextState
        if (currentState == endState):
            break

#Count length of path from Start to End      
currentState = 0
pathLength = [currentState]    
while True:
    nextState = np.argmax(Q[currentState,:])
    pathLength.append(nextState)
    currentState = nextState
    if (nextState == endState):
        break
        
print(Q)
print(len(pathLength)-1)
    


    

