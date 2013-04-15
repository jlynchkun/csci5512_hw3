"""
usage:
~/python-virtual-envs/2.7/csci5512/bin $ ./py.test ~/PycharmProjects/csci5512_hw3/

"""
import collections

import mdp
import problem_2

two_by_two_states = {(1, 1), (1, 2), (2, 1), (2, 2)}
two_by_two_actions = {'u', 'd', 'l', 'r'}
two_by_two_reward = {
    (1, 1): -0.1,
    (1, 2): -0.1,
    (2, 1): -0.1,
    (2, 2): 1.0
}
two_by_two_allowed_transitions = {
    (1, 1): {'d': (2, 1), 'r': (1, 2)},
    (1, 2): {'d': (2, 2), 'l': (1, 1)},
    (2, 1): {'u': (1, 1), 'r': (2, 2)},
    # (2, 2) is a terminal state
    (2, 2): {}
}
two_by_two_p_action_given_desired_action = {
    'u': {'u': 0.8, 'd': 0.0, 'l': 0.1, 'r': 0.1},
    'd': {'u': 0.0, 'd': 0.8, 'l': 0.1, 'r': 0.1},
    'l': {'u': 0.1, 'd': 0.1, 'l': 0.8, 'r': 0.0},
    'r': {'u': 0.1, 'd': 0.1, 'l': 0.0, 'r': 0.8}
}


def test_random_policy():
    m = mdp.MDP(
        states=two_by_two_states,
        actions=two_by_two_actions,
        reward=two_by_two_reward,
        allowed_transitions=two_by_two_allowed_transitions,
        p_action_given_desired_action=two_by_two_p_action_given_desired_action,
        gamma=0.0
    )
    for i in range(100):
        policy = m.get_random_policy()
        assert policy[(1, 1)] in [{'d': (2, 1)}, {'r': (1, 2)}]
        assert policy[(1, 2)] in [{'d': (2, 2)}, {'l': (1, 1)}]
        assert policy[(2, 1)] in [{'u': (1, 1)}, {'r': (2, 2)}]
        assert len(policy[(2, 2)].keys()) == 0


def test_evaluate_bellman_equation():
    m = mdp.MDP(
        states=two_by_two_states,
        actions=two_by_two_actions,
        reward=two_by_two_reward,
        allowed_transitions=two_by_two_allowed_transitions,
        p_action_given_desired_action=two_by_two_p_action_given_desired_action,
        gamma=0.3
    )
    U = {
        (1, 1): 0.0,
        (1, 2): 0.0,
        (2, 1): 0.0,
        (2, 2): 0.0
    }
    U_next = {state: 0.0 for state in m.states}
    # test a state with maximizing action moving to a non-terminal state
    (U_next[(1, 1)], maximizing_action) = m.evaluate_bellman_equation(U, (1, 1))
    assert U_next[(1, 1)] == (m.reward[(1, 1)] + 0.3 * (0.8 * U[(2, 1)] + 0.1 * U[(1, 2)] + 0.1 * U[(1, 1)]))
    #assert maximizing_action == 'd'
    # test a state with maximizing action moving to a terminal state
    (U_next[(1, 2)], maximizing_action) = m.evaluate_bellman_equation(U, (1, 2))
    assert U_next[(1, 2)] == m.reward[(1, 2)] + m.gamma * (0.8 * U[(2, 2)] + 0.1 * U[(1, 1)] + 0.1 * U[(1, 2)])
    #assert maximizing_action == 'd'
    # test a state with maximizing action moving to a terminal state
    (U_next[(2, 1)], maximizing_action) = m.evaluate_bellman_equation(U, (2, 1))
    assert U_next[(2, 1)] == m.reward[(2, 1)] + m.gamma * (0.8 * U[(2, 2)] + 0.1 * U[(1, 1)] + 0.1 * U[(2, 1)])
    # test the terminal state
    (U_next[(2, 2)], maximizing_action) = m.evaluate_bellman_equation(U, (2, 2))
    assert maximizing_action is None
    assert U_next[(2, 2)] == 1.0

    U = U_next.copy()
    print(U)
    # test a state with maximizing action moving to a non-terminal state
    (U_next[(1, 1)], maximizing_action) = m.evaluate_bellman_equation(U, (1, 1))
    assert U_next[(1, 1)] == (m.reward[(1, 1)] + m.gamma * (0.8 * U[(2, 1)] + 0.1 * U[(1, 2)] + 0.1 * U[(1, 1)]))
    assert maximizing_action == 'r'


def test_get_resulting_action_list():
    m = mdp.MDP(
        states=problem_2.states,
        actions=problem_2.actions,
        reward={},
        allowed_transitions=problem_2.allowed_transitions,
        p_action_given_desired_action=problem_2.p_action_given_desired_action,
        gamma=0.0
    )
    # from state (1, 1) and desired action 'u' we can move
    #   up with probability 0.8
    #   stay (left) with probability 0.1
    #   right with probability 0.1
    resulting_action_list = m.get_resulting_action_list((1, 1), 'u')
    assert len(resulting_action_list) == 3
    assert ('u', 0.8, (2, 1)) in resulting_action_list
    assert ('l', 0.1, (1, 1)) in resulting_action_list
    assert ('r', 0.1, (1, 2)) in resulting_action_list

    resulting_action_list = m.get_resulting_action_list((2, 1), 'u')
    assert len(resulting_action_list) == 3
    assert ('u', 0.8, (3, 1)) in resulting_action_list
    assert ('l', 0.1, (2, 1)) in resulting_action_list
    assert ('r', 0.1, (2, 1)) in resulting_action_list


# this test tries to reproduce the utility values from figure 17.3 R&N
def test_value_iteration():
    r = -0.04
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
        gamma=0.999999  # can't use 1.0
    )

    U, policy = m.value_iteration(0.001)
    assert_optimal_policy(U, policy)


def test_policy_iteration():
    r = -0.04
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
        gamma=0.999999  # can't use 1.0
    )

    U, policy = m.policy_iteration()
    assert_optimal_policy(U, policy)


def assert_optimal_policy(U, policy):
    assert policy[(1, 1)] == {'u': (2, 1)}
    assert 0.704 < U[(1, 1)] < 0.706

    assert policy[(1, 2)] == {'l': (1, 1)}
    assert 0.654 < U[(1, 2)] < 0.656

    assert policy[(1, 3)] == {'l': (1, 2)}
    assert 0.610 < U[(1, 3)] < 0.612

    assert policy[(1, 4)] == {'l': (1, 3)}
    assert 0.387 < U[(1, 4)] < 0.389

    assert policy[(2, 1)] == {'u': (3, 1)}
    assert 0.761 < U[(2, 1)] < 0.763

    assert len(policy[(2, 4)].keys()) == 0
    assert U[(2, 4)] == -1.0

    assert len(policy[(3, 4)].keys()) == 0
    assert U[(3, 4)] == 1.0


def test_policy_evaluation():
    m = mdp.MDP(
        states=two_by_two_states,
        actions=two_by_two_actions,
        reward=two_by_two_reward,
        allowed_transitions=two_by_two_allowed_transitions,
        p_action_given_desired_action=two_by_two_p_action_given_desired_action,
        gamma=0.3
    )

    # this policy only gets to the terminal state by accident
    policy = {
        (1, 1): {'d': (2, 1)},
        (1, 2): {'d': (2, 2)},
        (2, 1): {'u': (1, 1)},
        # (2, 2) is a terminal state
        (2, 2): {}
    }
    U = {state: 0.0 for state in two_by_two_states}
    U_update = m.policy_evaluation(policy, U)
    assert -0.125 < U_update[(1, 1)] < -0.124
    assert  0.140 < U_update[(1, 2)] <  0.141
    assert -0.103 < U_update[(2, 1)] < -0.102
    assert  U_update[(2, 2)] == 1.0
