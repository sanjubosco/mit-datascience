import numpy as np 

states = [0, 1, 2, 3, 4]
actions = [0, 1, 2]
V = np.zeros(5)
R = np.zeros(5)
R[4] = 1
T = np.array([ [[1/2,1/2,0,0,0], [1/2,1/2,0,0,0], [2/3,1/3,0,0,0]], [[1/3,2/3,0,0,0], [1/4,1/2,1/4,0,0], [0,2/3,1/3,0,0]], [[0,1/3,2/3,0,0], [0,1/4,1/2,1/4,0], [0,0,2/3,1/3,0]], [[0,0,1/3,2/3,0], [0,0,1/4,1/2,1/4], [0,0,0,2/3,1/3]], [[0,0,0,1/3,2/3], [0,0,0,1/2,1/2], [0,0,0,1/2,1/2]], ])
r = .5
steps = 10

num_states = len(states)
num_actions = len(actions)

for i in range(steps):
    Q = np.zeros((5,5))
    for s in range(num_states):
        for a in range(num_actions):
            for t in range (num_states):
                Q[s][a] += T[s][a][t] * (R[s] + r * V[t])

    V = np.max(Q, axis=1)

# for i in range(steps):
#     Q = [[sum([T[s][a][t] * (R[s] + r * V[t]) for t in range(num_states)]) for a in range(num_actions)] for s in range(num_states)]
#     print (Q)
#     V = np.max(Q, axis=1)
#     print("V is", V)

print(V)

