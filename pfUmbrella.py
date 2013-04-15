"""
pfUmbrella.py

Joshua Lynch
3471952
lynch197@umn.edu

usage: python pfUmbrella.py 100 FFFFFTTTTT

the first argument is numSamples, the second argument is the evidence

This program estimates the state of the final hidden rain variable given a sequence of observations of the umbrella
variable using particle filtering.  The file problem_1.py contains variables and functions used by lwUmbrella.py
and pfUmbrella.py.  Additionally this program calculates the exact filtered probabilities for the final rain variable.

"""
import bisect
import collections
import random
import sys


import problem_1


def particle_filtering_umbrella_network(numSamples, umbrella_evidence, rain_cpt, umbrella_cpt):
    # S is a list of N particles
    # each particle is 'T' or 'F'
    # initialize the particles with the prior probability of rain P(rain=T)=0.5
    S = [problem_1.sample_binary_random_variable('prior', rain_cpt) for n in range(numSamples)]
    W = [1.0] * numSamples
    for i, e in enumerate(umbrella_evidence):
        for n in range(numSamples):
            S[n] = problem_1.sample_binary_random_variable(S[n], rain_cpt)
            W[n] *= umbrella_cpt[S[n]][e]
        S, W = weighted_sample_with_replacement(S, W)

    return S


# this function resamples the input population S_0 (with corresponding weights W_0)
# to generate a new population S_1 with corresponding weights W_1
def weighted_sample_with_replacement(S_0, W_0):
    sum_W_0 = sum(W_0)
    W_0_normalized = [w_0 / sum_W_0 for w_0 in W_0]
    W_0_normalized_cumulative = []
    a = 0.0
    for w in W_0_normalized:
        a += w
        W_0_normalized_cumulative.append(a)
    # W_0_normalized_cumulative looks like [0.1, 0.23, 0.31, 0.56, 0.78, 0.91, 1.0]
    W_1 = [None] * len(W_0)
    S_1 = [None] * len(S_0)
    for i in range(len(W_0)):
        r_i = choose_weighted_sample_index(W_0_normalized_cumulative, random.random())
        W_1[i] = W_0[r_i]
        S_1[i] = S_0[r_i]
    # normalize the resampled weights
    sum_W_1 = sum(W_1)
    W_1_normalized = [w_1 / sum_W_1 for w_1 in W_1]
    return S_1, W_1_normalized


# return the index of the cumulative normalized weight nearest the floating point value in p
# using a binary search -- this is how we resample the population using each particle's weight
def choose_weighted_sample_index(cumulative_normalized_weights, p):
    return bisect.bisect(cumulative_normalized_weights, p)


def pfUmbrella():
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
        S = particle_filtering_umbrella_network(numSamples, umbrella_evidence, problem_1.rain_cpt, problem_1.umbrella_cpt)
        R_10_counts = collections.Counter()
        for s in S:
            R_10_counts[s] += 1
        p = {
            'T': float(R_10_counts['T'] / float(numSamples)),
            'F': float(R_10_counts['F'] / float(numSamples))
        }

        n += 1.0
        delta_T = p['T'] - mean_T
        mean_T += delta_T / n
        M2_T += delta_T * (p['T'] - mean_T)
        delta_F = p['F'] - mean_F
        mean_F += delta_F / n
        M2_F += delta_F * (p['F'] - mean_F)
    variance_T = M2_T / (n - 1)
    variance_F = M2_F / (n - 1)

    print('pf {} steps: p(R=T|{}) mean: {} variance: {} '.format(numSamples, umbrella_evidence, mean_T, variance_T))
    print('pf {} steps: p(R=F|{}) mean: {} variance: {} '.format(numSamples, umbrella_evidence, mean_F, variance_F))

    P_R_t_given_e_t = problem_1.filter_umbrella_network(umbrella_evidence, problem_1.rain_cpt, problem_1.umbrella_cpt)
    print('exact p(R=T|{}) = {}'.format(umbrella_evidence, P_R_t_given_e_t[0]))
    print('exact p(R=F|{}) = {}'.format(umbrella_evidence, P_R_t_given_e_t[1]))


if __name__ == '__main__':
    pfUmbrella()
