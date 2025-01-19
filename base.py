import numpy as np
import gymnasium
import minigrid
from collections import defaultdict
from dynamicobstacles import *
import matplotlib.pyplot as plt

alpha = 0.5 
gamma = 0.5 
episode = 1000 
epsilon = 0.1 


env = gymnasium.make("MiniGrid-Dynamic-Obstacles-5x5-v0") 

nbr_actions = int(env.action_space.n) 


def epsilon_greedy(env, state, Q, epsilon): 
    if np.random.rand() < epsilon: 
        action = np.random.choice(nbr_actions)
    else: 
        action = np.argmax(Q[state])
    
    return action

def Q_learning(alpha,epsilon,gamma):  
    
    Q = defaultdict(lambda: np.zeros(nbr_actions))
    fail = 0
    win = 0
    collision = 0
    reward_par_ep = []

    for i in range(episode):
        state = env.reset() 
        state = str(state)
        total_reward = 0

        while True: #atteint fin ou état impossible
            action = epsilon_greedy(env,state, Q,epsilon)
            next_state, reward,terminated,truncated,info = env.step(action) 
            next_state = str(next_state)

            #reward intermédiaire = si on fait une bonne action
            if reward == 0:
                reward += 0.01

            total_reward += reward
 
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action]) 

            if terminated:
                if reward == -1:
                    fail += 1
                    collision += 1
                else:
                    win += 1
                break
            elif truncated:
                fail += 1
                break

            state = next_state
        
        reward_par_ep.append(total_reward)

    print("Entrainement avec Q-LEARNING :")
    print(f"fails = {fail} et wins = {win}")
    print(f"Nombre collision avec obstacle = {collision}")
    print(f"Max step atteint : {fail - collision}")
    print(f"Taux de réussite : {(win/episode)*100} %")

    return Q, reward_par_ep

def SARSA(alpha,epsilon,gamma):

    Q = defaultdict(lambda: np.zeros(nbr_actions))
    reward_par_ep = []
    fail = 0
    win = 0
    collision = 0

    for i in range(episode):
        state = env.reset() 
        state = str(state)
        total_reward = 0
        action = epsilon_greedy(env,state, Q,epsilon)

        while True: #atteint fin ou état impossible
            next_state, reward,terminated,truncated,info = env.step(action)
            next_state = str(next_state)

            if reward == 0:
                reward += 0.01
            
            total_reward += reward

            next_action = epsilon_greedy(env,next_state, Q,epsilon)
  

            Q[state][action] = Q[state][action] + alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            if terminated:
                if reward == -1:
                    fail += 1
                    collision += 1
                else:
                    win +=1
                break
            elif truncated:
                fail +=1
                break
            
            state = next_state
            action = next_action
        
        reward_par_ep.append(total_reward)

    print("Entrainement avec SARSA :")
    print(f"fails = {fail} et wins = {win}")
    print(f"Nombre collision avec obstacle = {collision}")
    print(f"Max step atteint : {fail - collision}")
    print(f"Taux de réussite : {(win/episode)*100} %")

    return Q,reward_par_ep 

def affichage(Q):
    test = gymnasium.make("MiniGrid-Dynamic-Obstacles-5x5-v0", render_mode="human")
    state = test.reset()
    state = str(state)
    test.render()

    while True:
        action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, info = test.step(action)
        next_state = str(next_state)
        test.render()
        state = next_state

        if terminated == True:
            break

    test.close()
    

def graphe(tab):
    plt.figure(figsize=(12, 6)) 
    plt.plot(tab, label='Reward par épisode')

    plt.xlabel('Nombre d\'épisodes')
    plt.ylabel('Reward')
    plt.title('Reward par épisode')

    plt.xticks(range(0, len(tab), 100))
    plt.grid()

    plt.show()


def main():
    #Q, tab = Q_learning(alpha,epsilon,gamma)
    Q, tab = SARSA(alpha,epsilon,gamma)
    graphe(tab)
    #affichage(Q)

main()
env.close()



