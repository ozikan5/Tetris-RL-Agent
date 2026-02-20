import numpy as np
from collections import deque
import random
import torch
from tetris_rl.models.dqn import DQNModel
from tetris_rl.features import get_features
import torch.nn as nn
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, queue_len):
        # create the memory queue using dequeue with a fixed length
        self.memqueue = deque(maxlen=queue_len)

    # saves a specific experience inside the replay buffer
    def save(self, current_state_features, reward, next_state_features, game_over):
        # save the experience inside the buffer
        self.memqueue.append((current_state_features, reward, next_state_features, game_over))

    # selects random experiences from the replay buffer
    def recall(self, batch_size=64):
        # get random experiences from the buffer
        exps = random.sample(self.memqueue, k=batch_size)

        # create 4 different lists from the tuples
        states, rewards, next_states, game_overs = zip(*exps)

        # convert each group into its own strict PyTorch tensor
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        
        # game over is also a float for math purposes
        game_overs_t = torch.tensor(game_overs, dtype=torch.float32)

        return states_t, rewards_t, next_states_t, game_overs_t
    
    def size(self):
        return len(self.memqueue)
    
class DQNAgent:
    def __init__(self, batch_size=64, queue_len=100000, hidden_layer_size=64):
        self.learning_rate = 1e-3
        self.gamma = 0.98
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = batch_size

        self.buffer = ReplayBuffer(queue_len)
        self.model = DQNModel(hidden_layer_size)
        self.loss_fun = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # wraps board features inside a tensor
    def predict_value(self, features):
        features_t = torch.tensor(features, dtype=torch.float32)

        # no need to calculate gradients since we are just playing the game in this step
        with torch.no_grad():
            pred = self.model(features_t)
        
        # return the raw prediction
        return pred.item()
    
    # selects the best action given possible next states using the neural network
    def act(self, next_states):

        # if we choose epsilon
        if random.random() < self.epsilon:
            # just return a random aciton 
            return random.choice(list(next_states.keys()))
        
        # if we are doing greedy
        # initialize placeholders for best score and action
        best_score = -float('inf')
        best_action = None

        # iterate through possible next states and select the best action
        for action, (board, reward, game_over) in next_states.items():
            features = get_features(board)
            next_value = self.predict_value(features)
            
            # score = immediate reward + discounted future value (matches tabular agent)
            # no future value if game over
            if game_over:
                score = reward  
            else:
                score = reward + self.gamma * next_value

            if score > best_score:
                best_score = score
                best_action = action

        return best_action
    
    # function that applies Bellman logic using the replay buffer
    def learn(self):
        # if we don't have enough experiences in the buffer, skip the iteration
        if self.buffer.size() < self.batch_size:
            return
        
        # get the tensors
        states, rewards, next_states, game_overs = self.buffer.recall(self.batch_size)

        # we calculate the targets, that is the value of the current state from reward and value of next_state
        with torch.no_grad():
            # get our evaluation
            next_preds = self.model(next_states)

            # only reward remaining for finished games
            targets = rewards + (self.gamma * next_preds * (1.0 - game_overs))

        # now we have to get our predicions
        preds = self.model(states)

        # calculate our loss
        loss = self.loss_fun(preds, targets)

        # zero out optimizers memory
        self.optimizer.zero_grad()

        # calculate gradient for backpropogation
        loss.backward()

        # update weights of the model based on the loss
        self.optimizer.step()

    # function for decaying epsilon over time
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
