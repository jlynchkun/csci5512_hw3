"""
mdpVI.py

Joshua Lynch
3471952
lynch197@umn.edu

usage: python mdpVI.py -2.0

the first argument is the reward for non-terminal states

This program determines the optimal policy and associated state utilities for the Markov Decision Process described
in HW 3 Problem 2 using value iteration.  Most of the code for this program and mdpVI.py is contained in the file
mdp.py.  Common variables are defined in problem_2.py.

"""
import sys


import mdp
import problem_2


def mdpVI():
    r = float(sys.argv[1])

    m = mdp.MDP(
        states=problem_2.states,
        actions=problem_2.actions,
        reward={
            (3, 1): r, (3, 2): r, (3, 3): r, (3, 4):  1.0,
            (2, 1): r,            (2, 3): r, (2, 4): -1.0,
            (1, 1): r, (1, 2): r, (1, 3): r, (1, 4): r
        },
        allowed_transitions=problem_2.allowed_transitions,
        p_action_given_desired_action=problem_2.p_action_given_desired_action,
        gamma=0.99999
    )

    U, policy = m.value_iteration(0.001)
    for r in (3, 2, 1):
        for c in (1, 2, 3, 4):
            state = (r, c)
            if state in policy:
                print('policy for state {} is {} with utility {}'.format(state, policy.get(state, 'terminal'), U[state]))

if __name__ == '__main__':
    mdpVI()