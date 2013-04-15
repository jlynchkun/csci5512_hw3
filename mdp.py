"""
mdp.py

Markov Decision Process class definition used by mdpVI.py and mdpPI.py.
"""

import random


class MDP:
    def __init__(self, states, actions, reward, allowed_transitions, p_action_given_desired_action, gamma):
        self.states = states
        self.actions = actions
        self.reward = reward
        self.allowed_transitions = allowed_transitions
        self.p_action_given_desired_action = p_action_given_desired_action
        self.gamma = gamma

    # given a state and a desired action
    # return a list of possible resulting actions and the associated probabilities
    # for example if we are in state (2, 1) and our desired action is up:
    #   get_resulting_action_list((2, 1), 'u')
    # will return the list
    #   [('u', (3, 1), 0.8), ('l', (2, 1), 0.1), ('r', (2, 1), 0.1)]
    # which indicates we will go
    #   up to (3, 1) with probability 0.8
    #   left to (2, 1) with probability 0.1 -- no movement
    #   right to (2, 1) with probability 0.1 -- no movement
    def get_resulting_action_list(self, state, desired_action):
        resulting_action_list = []
        for resulting_action in self.p_action_given_desired_action[desired_action].keys():
            # the resulting action may not be allowed -- do not change state in this case
            # for example we can not go left from state (2, 1) so we will not move
            # assign the action 'left' probability of 0.1 with end point (2, 1) -- no actual movement
            p_resulting_action = self.p_action_given_desired_action[desired_action][resulting_action]
            if p_resulting_action > 0.0:
                if resulting_action in self.allowed_transitions[state].keys():
                    resulting_state = self.allowed_transitions[state][resulting_action]
                else:
                    # no movement
                    resulting_state = state
                resulting_action_list.append(
                    (resulting_action, p_resulting_action, resulting_state)
                )
            else:
                # this is the action opposite the desired action
                pass
        return resulting_action_list

    # given utilities for all states U and some state s
    # return the utility U(s) and maximizing action from
    #    U(s) = R(s) + gamma * max_over_a(sum_over_s'(T(s,a,s')U(s')))
    # optionally specify transitions=policy for simplified Bellman update:
    #    U_i+1 = R(s) + gamma * sum_over_s'(T(s,policy(s)),s')U_i(s'))
    def evaluate_bellman_equation(self, U, state, transitions=None):
        if transitions is None:
            transitions = self.allowed_transitions
        if len(transitions[state].keys()) == 0:
            # we are working with a terminal state
            return self.reward[state], None
        else:
            # we are working with a non-terminal state
            # initialize a dictionary to associate actions with sums over s' of T(s,a,s')U(s')
            # at the end we will choose the action with the maximum sum
            sum_for_desired_action = {desired_action: 0.0 for desired_action in transitions[state].keys()}
            for desired_action in transitions[state].keys():
                # desired_action is the action we want
                # resulting_action is the action we get -- same as s'
                resulting_action_list = self.get_resulting_action_list(state, desired_action)
                for resulting_action, p_resulting_action, resulting_state in resulting_action_list:
                    sum_for_desired_action[desired_action] += p_resulting_action * U[resulting_state]
            # select the desired action with maximum sum
            # Q: what happens if more than one desired_action gives the maximum utility?
            # A: the first of the desired_actions with maximum utility will be returned
            (maximizing_action, maximum_sum) = max(sum_for_desired_action.items(), key=lambda x: x[1])
            return self.reward[state] + self.gamma * maximum_sum, maximizing_action

    # this follows the value iteration method in Russell & Norvig
    def value_iteration(self, epsilon):
        delta = epsilon * (1.0 - self.gamma) / self.gamma
        # U and U_next associate a utility value with each state
        U = {state: 0.0 for state in self.states}
        U_next = {state: 0.0 for state in self.states}
        policy = {}
        step = 0
        while delta >= epsilon * (1.0 - self.gamma) / self.gamma:
            step += 1
            U = U_next.copy()
            delta = 0.0
            for state in self.states:
                U_next[state], maximizing_action = self.evaluate_bellman_equation(U, state)
                if maximizing_action is None:
                    # state is a terminal state
                    policy[state] = {}
                else:
                    policy[state] = {maximizing_action: self.allowed_transitions[state][maximizing_action]}
                delta_U = abs(U_next[state] - U[state])
                if delta_U > delta:
                    delta = delta_U
            #print('step: {} delta= {}'.format(step, delta))
        return U, policy

    # this is pretty much straight out of Russell & Norvig
    def policy_iteration(self):
        policy = self.get_random_policy()
        U = {state: 0.0 for state in self.states}
        unchanged = False
        while not unchanged:
            U = self.policy_evaluation(policy, U)
            unchanged = True
            for s in self.states:
                U_s, maximizing_action = self.evaluate_bellman_equation(U, s)
                U_policy, policy_action = self.evaluate_bellman_equation(U, s, policy)
                if U_s > U_policy:
                    end_state = self.allowed_transitions[s][maximizing_action]
                    policy[s] = {maximizing_action: end_state}
                    unchanged = False
        return U, policy

    def get_random_policy(self):
        # a policy associates a desired action and corresponding end point with each state
        # a random policy selects one action and corresponding end point for each state
        random_policy = {}
        for state in self.states:
            if len(self.allowed_transitions[state].keys()) == 0:
                # state is terminal
                random_policy[state] = {}
            else:
                random_action = random.choice(self.allowed_transitions[state].keys())
                random_policy[state] = {random_action: self.allowed_transitions[state][random_action]}
        return random_policy

    # you got me
    # I used modified policy iteration
    def policy_evaluation(self, policy, U):
        U_1 = U.copy()
        for k in range(10):
            U_0 = U_1.copy()
            for state in U_0:
                U_1[state], action = self.evaluate_bellman_equation(U_0, state, policy)
        return U_1
