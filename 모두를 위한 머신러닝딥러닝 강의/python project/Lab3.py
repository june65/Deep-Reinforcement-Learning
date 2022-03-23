import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
    m=np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4', 'is_slippery':False}

)

env = gym.make('FrozenLake-v3')

Q= np.zeros([env.observation_space.n,env.action_space.n])
num_epsodes = 100

rList = []
for i in range(num_epsodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:

        action = rargmax(Q[state, :])

        new_state, reward, done,_ = env.step(action)

        Q[state,action] = reward + np.max(Q[new_state,:])

        rAll+= reward
        state = new_state
    rList.append(rAll)


print("success rate:" + str(sum(rList)/num_epsodes))
print("Final Q Table Values")
print("LEFt DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)),rList,color="blue")
plt.show()