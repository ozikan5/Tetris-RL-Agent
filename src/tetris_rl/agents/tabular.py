import numpy as np
import random
from tetris_rl.features import get_features
from collections import defaultdict

class TabularAgent:
    def __init__(self):
        
        # a dictionary mapping states to a Q-value
        self.q_table = defaultdict(float)
        
        # hyperparameters for learning (will be decayed in training)
        
        # was previously 0.1
        self.learning_rate = 0.2
        # was previously 0.9
        self.gamma = 0.98
        # was previously 0.1
        self.epsilon = 0.5

    
    # given the features, this function assigns values into buckets for easier state mapping
    # return a tuple of 4 buckets (finer resolution for critical features)
    def discretize(self, features):
        agg_height = features[0]
        holes = features[1]
        bumpiness = features[2]
        max_height = features[3]

        # increased bucket sizes to capture more detail in critical features
        agg_bucket = min(7, int(agg_height / 15))
        holes_bucket = min(7, int(holes))
        bump_bucket = min(6, int(bumpiness / 4))
        max_h_bucket = min(7, int(max_height / 3))

        return (agg_bucket, holes_bucket, bump_bucket, max_h_bucket)
    
    # selects the best action given the possible next_states using epsilon-greedy strategy
    def select_action(self, next_states):
        
        # if we are selecting at random as part of epsilon
        if random.random() < self.epsilon:
            # select a random next_state
            return random.choice(list(next_states.keys()))
        
        # if we didn't hit the epsilon, we are doing greedy

        # initialize best score to negative infinity and aciton to none
        best_score = -float('inf')
        best_action = None

        # for every action and possible results following
        for action, (board, reward, game_over) in next_states.items():
            
            # get the feature from the board
            features = get_features(board)
            
            # assign features to buckets
            buckets = self.discretize(features)
            
            # score = immediate reward + discounted value of next state
            score = reward + self.gamma * self.q_table.get(buckets, 0.0)
            
            if score > best_score:
                best_score = score
                best_action = action

        return best_action
    
    # Updates V(state) using TD(0): V(s) <- V(s) + lr * (r + gamma * V(s') - V(s))
    # current_features: state we were in when we took the action
    # next_state_features: state we landed in after the action (used for bootstrap)
    def update(self, current_features, reward, next_state_features, game_over):
        # discretize the current features
        feature_buckets = self.discretize(current_features)
        # look up the value of the current state
        value = self.q_table.get(feature_buckets, 0.0)

        if game_over:
            # if the game is over, the value of the next state is 0
            next_value = 0.0
        else:
            next_buckets = self.discretize(next_state_features)
            next_value = self.q_table.get(next_buckets, 0.0)

        # calculate the TD target
        td_target = reward + self.gamma * next_value
        # update the new Q-score using Bellman equation
        new_q_score = value + self.learning_rate * (td_target - value)
        self.q_table[feature_buckets] = new_q_score
