import numpy as np
import random
from tetris_features import get_features

class TabularAgent:
    def __init__(self):
        
        # a dictionary mapping states to a Q-value
        self.q_table = {} 
        
        # hyperparameters for learning 
        self.learning_rate = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    
    # given the features, this function assigns values into buckets for easier state mapping
    # return a tuple of 4 buckets
    def discretize(self, features):
        
        agg_height = features[0]
        holes = features[1]
        bumpiness = features[2]
        max_height = features[3]

        agg_bucket = min(5, agg_height // 10)
        holes_bucket = min(5, holes)
        bump_bucket = min(5, bumpiness // 5)
        max_h_bucket = min(5, max_height // 5)

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
            
            # look up if we recorded a score for this state before
            # if not initialize it with 0
            # also add the reward
            score = reward + self.q_table.get(buckets, 0.0)
            
            if score > best_score:
                best_score = score
                best_action = action

        return best_action
    
    # updates the Q-table using Bellman equation
    def update(self, current_features, reward, next_states, game_over):
        feature_buckets = self.discretize(current_features)

        value = self.q_table.get(feature_buckets, 0.0)

        max_next_q = 0.0

        if not game_over and next_states:
            max_next_q = -float('inf')
            
            for action, (board, next_reward, next_game_over) in next_states.items():

                # get the feature from the board
                features = get_features(board)
                
                # assign features to buckets
                buckets = self.discretize(features)
                
                # look up if we recorded a score for this state before
                # if not initialize it with 0
                # also add the reward
                score = next_reward + self.q_table.get(buckets, 0.0)

                max_next_q = max(max_next_q, score)

        new_q_score = value + self.learning_rate * (reward + self.gamma * max_next_q - value)
        self.q_table[feature_buckets] = new_q_score

