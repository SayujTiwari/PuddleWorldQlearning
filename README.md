# PuddleWorldQlearning

took reference from https://github.com/udacity/deep-reinforcement-learning/blob/master/discretization/Discretization_Solution.ipynb

A Q-learning implementation for the PuddleWorld environment where an agent learns to navigate from a start point to a goal while avoiding puddles. The agent uses a discretized state space and learns optimal actions through experience.
How it Works
The Q-learning algorithm works by:

Discretizing the continuous state space into a grid
Maintaining a Q-table that stores state-action values
Using an epsilon-greedy strategy for exploration/exploitation
Learning from experience by updating Q-values based on rewards received

Key Components:

State Space: Continuous 2D environment discretized into a grid
Actions: Agent can move in discrete directions
Rewards: Negative rewards for puddle contact, positive reward for reaching goal
Learning Parameters:

Learning Rate (α): 0.02
Discount Factor (γ): 0.99
Initial Exploration Rate (ε): 1.0
Exploration Decay: 0.9995

The agent gradually learns optimal paths by balancing exploration of new actions with exploitation of known good actions, eventually finding efficient routes to the goal while avoiding puddle.
