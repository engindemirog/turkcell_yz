import random

#environment
grid_size = 10
goal_state = (5,3)
actions = ["up","down","left","right"]

#q_table
q_table = {}

for x in range(grid_size):
    for y in range(grid_size):
        q_table[(x,y)] = [0,0,0,0]

#hiperparametreler
learning_rate = 0.2 #yeni öğrenilen etkisi
discount_factor = 0.9 #şimdi mi önemli sonra mı?
epsilon = 0.9 #yeni deneyimler edinme oranı
epochs = 500 #tecrübe


def get_next_state(state,action):
    x,y = state
    if action =="up":
        x = max(x-1,0)
    elif action=="down":
        x = min(x+1,grid_size-1)
    if action =="left":
        y = max(y-1,0)
    elif action=="right":
        y = min(y+1,grid_size-1)
    return (x,y)

for epoch in range(epochs):
    state = (0,0)
    while state != goal_state:
        if random.random()<epsilon:
            action = random.choice(actions)
        else:
            action_values = q_table[state]
            best_idx = action_values.index(max(action_values))
            action = actions[best_idx]
        
        next_state = get_next_state(state,action)
        reward = 1 if next_state == goal_state else -1

        #q tablosunu güncelle
        action_idx = actions.index(action)
        best_future = max(q_table[next_state])
        q_table[state][action_idx] += learning_rate * (reward + discount_factor * best_future - q_table[state][action_idx])

        state = next_state
    
        
        
    epsilon = max(0.01,epsilon*0.99)


for x in range(grid_size):
    for y in range(grid_size):
        values = q_table[(x, y)]
        print(f"State ({x},{y}): Up={values[0]:.2f}, Down={values[1]:.2f}, Left={values[2]:.2f}, Right={values[3]:.2f}")