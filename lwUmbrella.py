"""
lwUmbrella.py

Joshua Lynch
3471952
lynch197@umn.edu

usage: python lwUmbrella.py 100 FFFFFTTTTT

the first argument is numSamples, the second argument is the evidence

This program estimates the state of the final hidden rain variable given a sequence of observations of the umbrella
variable using likelihood weighting.  The file problem_1.py contains variables and functions used by lwUmbrella.py
and pfUmbrella.py.  Additionally this program calculates the exact filtered probabilities for the final rain variable.
"""

import sys


import problem_1


def likelihood_weighting_umbrella_network(numSamples, umbrella_evidence, rain_cpt, umbrella_cpt):
    # accumulate W['T'] and W['F']
    W = {'T': 0.0, 'F': 0.0}
    for i in range(numSamples):
        x, w = weighted_sample_umbrella_network(umbrella_evidence, rain_cpt, umbrella_cpt)
        # the last element of x is rain_10
        rain_end = x[-1]
        W[rain_end] = (W[rain_end] + w)
    return normalize(W)


# this function returns a random sample from the umbrella network (a string of Ts and 
# Fs representing the states of the rain variables) and the associated likelihood weight
def weighted_sample_umbrella_network(umbrella_evidence, rain_cpt, umbrella_cpt):
    # return sample x and weight w
    x = []
    w = 1.0
    # sample rain for t=0
    rain_0 = problem_1.sample_binary_random_variable('prior', rain_cpt)
    w *= weight_for_evidence(rain_0, umbrella_cpt, umbrella_evidence[0])
    x.append(rain_0)
    # sample rain for t>0
    for t in range(1, len(umbrella_evidence)):
        # sample rain_t
        rain_t_minus_1 = x[t - 1]
        rain_t = problem_1.sample_binary_random_variable(rain_t_minus_1, rain_cpt)
        x.append(rain_t)
        # update the weight
        w *= weight_for_evidence(rain_t, umbrella_cpt, umbrella_evidence[t])

    return x, w


def normalize(P):
    Z = sum(P.values())
    return {k: p / Z for k, p in P.items()}


def weight_for_evidence(parent, cpt, evidence):
    return cpt[parent][evidence]


def lwUmbrella():
    numSamples = int(sys.argv[1])  # 100 or 1000
    umbrella_evidence = sys.argv[2]  # 'TTTTTFFFFF'

    # repeat the calculation several times to get variance
    # using online mean and variance calculation
    n = 0.0
    mean_T = 0.0
    mean_F = 0.0
    M2_T = 0.0
    M2_F = 0.0
    for i in range(1000):
        p = likelihood_weighting_umbrella_network(numSamples, umbrella_evidence, problem_1.rain_cpt, problem_1.umbrella_cpt)
        n += 1.0
        delta_T = p['T'] - mean_T
        mean_T += delta_T / n
        M2_T += delta_T * (p['T'] - mean_T)
        delta_F = p['F'] - mean_F
        mean_F += delta_F / n
        M2_F += delta_F * (p['F'] - mean_F)
    variance_T = M2_T / (n - 1)
    variance_F = M2_F / (n - 1)
    #print('{} steps: p(R|{}) = {}'.format(numSamples, umbrella_evidence, p))
    print('lw {} steps: p(R=T|{}) mean: {} variance: {} '.format(numSamples, umbrella_evidence, mean_T, variance_T))
    print('lw {} steps: p(R=F|{}) mean: {} variance: {} '.format(numSamples, umbrella_evidence, mean_F, variance_F))

    P_R_t_given_e_t = problem_1.filter_umbrella_network(umbrella_evidence, problem_1.rain_cpt, problem_1.umbrella_cpt)
    print('exact p(R=T|{}) = {}'.format(umbrella_evidence, P_R_t_given_e_t[0]))
    print('exact p(R=F|{}) = {}'.format(umbrella_evidence, P_R_t_given_e_t[1]))


if __name__ == '__main__':
    lwUmbrella()