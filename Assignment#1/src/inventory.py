import numpy as np
import matplotlib.pyplot as plt

np.printoptions(precision=3) 

num_states = 11 # s0 .. s1... s11
num_actions = 2 # 0=> Buy, 1=> Sell

R = np.zeros((num_states,num_actions))
T = np.zeros((num_states,num_actions,num_states))

# Rewards
#TODO: Confirm the R[9,0] = 100
R[9,0] = 100
for s in range(2, num_states-1):
   R[s,1] = 1

T[0,0,1] = 1
T[1,0,2] = 1
T[2,0,3] = 1
T[3,0,4] = 1
T[4,0,5] = 1
T[5,0,6] = 1
T[6,0,7] = 1
T[7,0,8] = 1
T[8,0,9] = 1
T[9,0,10] = 1
#T[10,0,10] = 0
#T[10,1,9] = 0
T[9,1,8] = 1
T[8,1,7] = 1
T[7,1,6] = 1
T[6,1,5] = 1
T[5,1,4] = 1
T[4,1,3] = 1
T[3,1,2] = 1
T[2,1,1] = 1
T[1,1,0] = 1
#T[0,1,0] = 1


#print("R:", R)
#print("T:", T)


""" for action in range(num_actions):
    plt.imshow(T[:, action, :], cmap='viridis', aspect='auto')
    plt.title(f'Transition matrix for action {action}')
    plt.xlabel('Next State')
    plt.ylabel('Current State')
    plt.colorbar(label='Probability')
    plt.xticks(np.arange(num_states))  # Set x-axis ticks at every state
    plt.yticks(np.arange(num_states))  # Set y-axis ticks at every state
    plt.show() """

gamma = 0.6
tol = 1e-3


def value_iteration(T,R,gamma,tol):
   V = np.zeros(num_states)
   policy = np.zeros(num_states, dtype=int)
   diff = np.inf
   index = 0
   while diff > tol:
       index += 1
       V_new = R + gamma * np.sum(T * V, axis=2)
       BV = np.max(V_new, axis=1)
       diff = np.linalg.norm(BV - V, ord=np.inf)
       V = BV
   policy = np.argmax(V_new, axis=1)
   print("Converged after", index, "iterations")
   return V, policy

def finite_horizon(H,T,R,gamma):
    seq = "s3"
    reward = 0
    s_old = 3
    s_new = 3
    t = 0
    while t < H:
        a_rand = np.random.choice(num_actions)

        # Determine the new state based on the action taken
        if(a_rand == 0):
            # Buy action
            if(s_new < num_states - 1):
                s_new = s_new + 1
        else:
            # Sell action
            if(s_new > 0):
                s_new = s_new - 1
        

        #print("Time step:", t)
        #print("Action chosen:", a_rand)
        #print("New state:", s_new)
        seq += f" -> s{s_new}"
        reward = reward + (gamma * R[s_old, a_rand])
        s_old = s_new
        t = t + 1
      
        if s_new == 10:
            break
        
    print("Sequence:", seq)
    print("Total Reward:", reward)

if __name__ == "__main__":
   gamma = 1.0
   V, policy = value_iteration(T,R,gamma,tol)
   print("Value Function V:", V)
   print("Optimal Policy:", policy)

   i = 0
   while i < 10:
      i += 1
      finite_horizon(11, T, R, gamma)