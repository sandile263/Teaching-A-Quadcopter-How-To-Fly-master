#!/usr/bin/env python3
# -*- coding: utf-8 -*-


##############################################
# PROGRAMMER: Udacity, Inc.                  #
# DATE CREATED: 09/05/2019                   #
# REVISED DATE: -                            #
# PURPOSE: This file contains a sample agent #
##############################################


##################
# Needed imports #
##################

import numpy as np
from task import Task


############################
# Class PolicySearch Agent #
############################

class PolicySearch_Agent():
    
    # Initialization:
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        self.w = np.random.normal(size=(self.state_size, self.action_size),
                                  scale=(self.action_range/(2*self.state_size)))
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1
        self.reset_episode()

    # Reset episode:
    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    # Save experience/reward and learn at end of episode:
    def step(self, reward, done):
        self.total_reward += reward
        self.count += 1
        if done:
            self.learn()

    # Choose action based on given state and policy:
    def act(self, state):
        action = np.dot(state, self.w)
        return action

    # Learn by random policy search using a reward-based score:
    def learn(self):
        self.score = self.total_reward/float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5*self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0*self.noise_scale, 3.2)
        # Equal noise in all directions:
        self.w = self.w + self.noise_scale*np.random.normal(size=self.w.shape)