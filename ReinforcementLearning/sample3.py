import random

# Ortam parametreleri
grid_size = 10
goal_state = (5, 3)
actions = ["up", "down", "left", "right"]

# Q-tablosu: Her konum için 4 aksiyon değeri
q_table = {(x, y): [0.0, 0.0, 0.0, 0.0] for x in range(grid_size) for y in range(grid_size)}

# Hiperparametreler
learning_rate = 0.2
discount_factor = 0.9
epsilon = 0.9
epochs = 500

# Bir sonraki durumu hesaplayan fonksiyon
def get_next_state(state, action):
    x, y = state
    if action == "up":
        x = max(x - 1, 0)
    elif action == "down":
        x = min(x + 1, grid_size - 1)
    elif action == "left":
        y = max(y - 1, 0)
    elif action == "right":
        y = min(y + 1, grid_size - 1)
    return (x, y)

# Q-learning döngüsü
for epoch in range(epochs):
    state = (0, 0)
    
    while state != goal_state:
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            best_idx = q_table[state].index(max(q_table[state]))
            action = actions[best_idx]

        next_state = get_next_state(state, action)
        reward = 1 if next_state == goal_state else -1

        action_idx = actions.index(action)
        best_future = max(q_table[next_state])
        current_value = q_table[state][action_idx]

        # Q-değeri güncellemesi
        q_table[state][action_idx] = current_value + learning_rate * (
            reward + discount_factor * best_future - current_value
        )

        state = next_state

    # Epsilon azalt (keşif oranı azalıyor)
    epsilon = max(0.01, epsilon * 0.99)

# Q tablosunu yazdır
for x in range(grid_size):
    for y in range(grid_size):
        up, down, left, right = q_table[(x, y)]
        print(f"State ({x},{y}): Up={up:.2f}, Down={down:.2f}, Left={left:.2f}, Right={right:.2f}")