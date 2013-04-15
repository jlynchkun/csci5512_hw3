"""
problem_1.py
"""

import random


# The cpt for the rain variable:
# {
#    R_t_minus_1 = T: { R_t = T: p,   R_t = F: 1 - p},
#    R_t_minus_1 = F: { R_t = T: p,   R_t = F: 1 - p},
#    'prior'        : { R_t = T: 1/2  R_t = F: 1/2  }
# }
# P(R_t|R_t_minus_1) is given by rain_cpt[R_t_minus_1][R_t]
rain_cpt = {
    'T':     {'T': 0.7, 'F': 0.3},
    'F':     {'T': 0.3, 'F': 0.7},
    'prior': {'T': 0.5, 'F': 0.5}
}

# the cpt for the umbrella variable is similar
# P(U_t|R_t) is given by umbrella_cpt[R_t][U_t]
umbrella_cpt = {
    'T': {'T': 0.9, 'F': 0.1},
    'F': {'T': 0.2, 'F': 0.8}
}


def sample_binary_random_variable(parent, cpt):
    P = cpt[parent]
    u = random.random()
    if u < P['T']:
        return 'T'
    else:
        return 'F'


def filter_umbrella_network(umbrella_evidence, rain_cpt, umbrella_cpt):
    # calculate the exact probabilities
    # on day one...
    P_R_t = normalize_vector(
        vector_sum(
            scalar_product((rain_cpt['T']['T'], rain_cpt['T']['F']), rain_cpt['prior']['T']),
            scalar_product((rain_cpt['F']['T'], rain_cpt['F']['F']), rain_cpt['prior']['F'])
        )
    )
    P_R_t_given_e_t = []
    # using evidence
    for e in umbrella_evidence:
        #print('P_R_t: {}'.format(P_R_t))
        P_R_t_given_e_t = normalize_vector(
            elementwise_product(
                (umbrella_cpt['T'][e], umbrella_cpt['F'][e]),
                P_R_t
            )
        )
        #print('P_R_t_given_e_t: {}'.format(P_R_t_given_e_t))
        P_R_t = normalize_vector(
            vector_sum(
                scalar_product((rain_cpt['T']['T'], rain_cpt['T']['F']), P_R_t_given_e_t[0]),
                scalar_product((rain_cpt['F']['T'], rain_cpt['F']['F']), P_R_t_given_e_t[1])
            )
        )
    return P_R_t_given_e_t


def scalar_product(v, s):
    return [v_i * s for v_i in v]


def vector_sum(v, w):
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def normalize_vector(v):
    return [v_i / sum(v) for v_i in v]


def elementwise_product(v, w):
    return [v_i * w_i for v_i, w_i in zip(v, w)]


