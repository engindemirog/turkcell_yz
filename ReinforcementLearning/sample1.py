import random

#environment
n_states = 10
goal_state = 6
actions = ["left","right"]

#q_table
q_table = [[0,0] for _ in range(n_states)]

#hiperparametreler
learning_rate = 0.1 #yeni öğrenilen etkisi
discount_factor = 0.9 #şimdi mi önemli sonra mı?
epsilon = 0.2 #yeni deneyimler edinme oranı
epochs = 100 #tecrübe

for epoch in range(epochs):
    state = 0
    while state != goal_state:
        if random.random()<epsilon:
            action = random.choice(actions)
        else:
            action = actions[q_table[state].index(max(q_table[state]))] # en iyi aksiyon
        
        if action == "right":
            next_state = min(state+1, n_states-1)
        else:
            next_state = max(state-1,0)
        
        reward = 1 if next_state == goal_state else -1

        #q tablosunu güncelle
        action_idx = actions.index(action)
        best_future = max(q_table[next_state])
        q_table[state][action_idx] += learning_rate * (reward + discount_factor * best_future - q_table[state][action_idx])

        state = next_state
    
        for i, row in enumerate(q_table):
            print(f"State {i}: Left={row[0]:.2f}, Right={row[1]:.2f}")


# 10*10 