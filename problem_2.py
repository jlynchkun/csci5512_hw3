"""
problem_2.py
"""

# these are the states for the MDP
states = {
    (3, 1), (3, 2), (3, 3), (3, 4),
    (2, 1),         (2, 3), (2, 4),
    (1, 1), (1, 2), (1, 3), (1, 4)
}

# these are the actions
actions = {'u', 'd', 'l', 'r'}

# these are the results for each action from each state
allowed_transitions = {
    #        up           down         left         right
    (3, 1): {             'd': (2, 1),              'r': (3, 2)},
    (3, 2): {                          'l': (3, 1), 'r': (3, 3)},
    (3, 3): {             'd': (2, 3), 'l': (3, 2), 'r': (3, 4)},
    (3, 4): {},
    (2, 1): {'u': (3, 1), 'd': (1, 1)                          },
    # there is no (2, 2) state
    (2, 3): {'u': (3, 3), 'd': (1, 3),              'r': (2, 4)},
    (2, 4): {},
    (1, 1): {'u': (2, 1),                           'r': (1, 2)},
    (1, 2): {                          'l': (1, 1), 'r': (1, 3)},
    (1, 3): {'u': (2, 3),              'l': (1, 2), 'r': (1, 4)},
    (1, 4): {'u': (2, 4),              'l': (1, 3)             }
}

# these are the probabilities of each result given an action
p_action_given_desired_action = {
    'u': {'u': 0.8, 'd': 0.0, 'l': 0.1, 'r': 0.1},
    'd': {'u': 0.0, 'd': 0.8, 'l': 0.1, 'r': 0.1},
    'l': {'u': 0.1, 'd': 0.1, 'l': 0.8, 'r': 0.0},
    'r': {'u': 0.1, 'd': 0.1, 'l': 0.0, 'r': 0.8}
}


