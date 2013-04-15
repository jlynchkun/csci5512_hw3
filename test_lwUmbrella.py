"""
usage:
~/python-virtual-envs/2.7/csci5512/bin $ ./py.test ~/PycharmProjects/csci5512_hw3/

"""

import problem_1
import lwUmbrella


def test_normalize():
    W = {'T': 1.0, 'F': 1.0}
    normalized_W = lwUmbrella.normalize(W)
    assert len(normalized_W.keys()) == 2
    assert normalized_W['T'] == 1.0 / (1.0 + 1.0)
    assert normalized_W['F'] == 1.0 / (1.0 + 1.0)


def test_sample_binary_random_variable():
    cpt = {'T': {'T': 1.0, 'F': 0.0}, 'F': {'T': 0.0, 'F': 1.0}, 'prior': {'T': 1.0, 'F': 0.0}}
    for i in range(100):
        s = problem_1.sample_binary_random_variable('prior', cpt)
        assert s == 'T'
        s = problem_1.sample_binary_random_variable('T', cpt)
        assert s == 'T'
        s = problem_1.sample_binary_random_variable('F', cpt)
        assert s == 'F'


def test_weight_for_evidence():
    cpt = {'T': {'T': 1.0, 'F': 0.0}, 'F': {'T': 0.0, 'F': 1.0}}
    assert lwUmbrella.weight_for_evidence('T', cpt, 'T') == 1.0
    assert lwUmbrella.weight_for_evidence('T', cpt, 'F') == 0.0
    assert lwUmbrella.weight_for_evidence('F', cpt, 'T') == 0.0
    assert lwUmbrella.weight_for_evidence('F', cpt, 'F') == 1.0


def test_weighted_sample_umbrella_network():
    rain_cpt = {'T': {'T': 1.0, 'F': 0.0}, 'F': {'T': 0.0, 'F': 1.0}, 'prior': {'T': 1.0, 'F': 0.0}}
    umbrella_cpt = {'T': {'T': 0.8, 'F': 0.2}, 'F': {'T': 0.3, 'F': 0.7}}

    umbrella_evidence = 'T'
    x, w = lwUmbrella.weighted_sample_umbrella_network(umbrella_evidence, rain_cpt, umbrella_cpt)
    assert x == ['T']
    assert w == 1.0 * 0.8

    umbrella_evidence = 'F'
    x, w = lwUmbrella.weighted_sample_umbrella_network(umbrella_evidence, rain_cpt, umbrella_cpt)
    assert x == ['T']
    assert w == 1.0 * 0.2

    umbrella_evidence = 'TT'
    x, w = lwUmbrella.weighted_sample_umbrella_network(umbrella_evidence, rain_cpt, umbrella_cpt)
    assert x == ['T', 'T']
    assert w == 1.0 * 0.8 * 0.8

    umbrella_evidence = 'FF'
    x, w = lwUmbrella.weighted_sample_umbrella_network(umbrella_evidence, rain_cpt, umbrella_cpt)
    assert x == ['T', 'T']
    assert w == 1.0 * 0.2 * 0.2

    umbrella_evidence = 'TF'
    x, w = lwUmbrella.weighted_sample_umbrella_network(umbrella_evidence, rain_cpt, umbrella_cpt)
    assert x == ['T', 'T']
    assert w == 1.0 * 0.8 * 0.2


def test_likelihood_weighting_umbrella_network():
    rain_cpt = {'T': {'T': 1.0, 'F': 0.0}, 'F': {'T': 0.0, 'F': 1.0}, 'prior': {'T': 1.0, 'F': 0.0}}
    umbrella_cpt = {'T': {'T': 1.0, 'F': 0.0}, 'F': {'T': 0.0, 'F': 1.0}}

    N = 1
    umbrella_evidence = 'T'
    normalized_W = lwUmbrella.likelihood_weighting_umbrella_network(N, umbrella_evidence, rain_cpt, umbrella_cpt)
    assert normalized_W['T'] == 1.0
    assert normalized_W['F'] == 0.0

    N = 1
    umbrella_evidence = 'TT'
    normalized_W = lwUmbrella.likelihood_weighting_umbrella_network(N, umbrella_evidence, rain_cpt, umbrella_cpt)
    assert normalized_W['T'] == 1.0
    assert normalized_W['F'] == 0.0

    N = 2
    umbrella_evidence = 'T'
    normalized_W = lwUmbrella.likelihood_weighting_umbrella_network(N, umbrella_evidence, rain_cpt, umbrella_cpt)
    assert normalized_W['T'] == 1.0
    assert normalized_W['F'] == 0.0

    N = 2
    umbrella_evidence = 'TT'
    normalized_W = lwUmbrella.likelihood_weighting_umbrella_network(N, umbrella_evidence, rain_cpt, umbrella_cpt)
    assert normalized_W['T'] == 1.0
    assert normalized_W['F'] == 0.0
