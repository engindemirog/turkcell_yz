import random

#environment variables
n_states = 10
goal_state = 9
actions = ["left","right"]

#q table
q_table = [[0,0] for _ in range(n_states)] #Hafıza, deneyim defteri

#hiperparametreler
learning_rate = 0.1 #yeni öğrenilen etkisi
discount_factor = 0.9 #şimdi mi önemli sonra mı 
epsilon = 0.9 #yeni deneyimler edinme oranı
epochs = 100 #tecrübe


for epoch in range(epochs):
    state = 0
    step = 0
    max_steps = 100
    while state!=goal_state and step<max_steps:
        if random.random()<epsilon: #keşif yap
            action = random.choice(actions)
            #print(action)
        else:
            action = actions[q_table[state].index(max(q_table[state]))] # en iyi aksiyon
        
        if action == "right":
            next_state = min(state + 1,n_states-1)
        else:
            next_state = max(state - 1,0)

        reward = 1 if next_state ==goal_state else -1

        #q tablosunu güncelle
        action_idx = actions.index(action)
        best_future = max(q_table[next_state])
        q_table[state][action_idx] += learning_rate * (reward + discount_factor * best_future - q_table[state][action_idx])

        state = next_state
        step +=1
    
    epsilon = max(0.01,epsilon*0.99)
    print(epsilon)

for i, row in enumerate(q_table):
    print(f"State {i}: Left={row[0]:.2f}, Right={row[1]:.2f}")


#10*10