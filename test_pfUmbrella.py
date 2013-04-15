"""
usage:
~/python-virtual-envs/2.7/csci5512/bin $ ./py.test ~/PycharmProjects/csci5512_hw3/

"""

import pfUmbrella


def test_choose_weighted_sample_index():
    weights = [0.1, 0.2, 0.3, 0.4]
    normalized_weights = [w / sum(weights) for w in weights]
    cumulative_normalized_weights = [sum(normalized_weights[0:(i+1)]) for i in range(len(normalized_weights))]
    # cumulative_normalized_weights should look like [0.1, 0.3, 0.6, 1.0]
    assert cumulative_normalized_weights == [0.1, 0.1 + 0.2, 0.1 + 0.2 + 0.3, 0.1 + 0.2 + 0.3 + 0.4]
    print('cumulative normalized weights: {}'.format(cumulative_normalized_weights))
    assert pfUmbrella.choose_weighted_sample_index(cumulative_normalized_weights, 0.0) == 0
    assert pfUmbrella.choose_weighted_sample_index(cumulative_normalized_weights, 0.09) == 0
    assert pfUmbrella.choose_weighted_sample_index(cumulative_normalized_weights, 0.1) == 1
    assert pfUmbrella.choose_weighted_sample_index(cumulative_normalized_weights, 0.11) == 1
    assert pfUmbrella.choose_weighted_sample_index(cumulative_normalized_weights, 0.29) == 1
    assert pfUmbrella.choose_weighted_sample_index(cumulative_normalized_weights, 0.30) == 1
    assert pfUmbrella.choose_weighted_sample_index(cumulative_normalized_weights, 0.31) == 2

    assert pfUmbrella.choose_weighted_sample_index(cumulative_normalized_weights, 0.59) == 2
    assert pfUmbrella.choose_weighted_sample_index(cumulative_normalized_weights, 0.60) == 2
    assert pfUmbrella.choose_weighted_sample_index(cumulative_normalized_weights, 0.61) == 3
    assert pfUmbrella.choose_weighted_sample_index(cumulative_normalized_weights, 0.99) == 3

    # this should not happen since the second argument should be in the range [0.0, 1.0)
    assert pfUmbrella.choose_weighted_sample_index(cumulative_normalized_weights, 1.0) == 4
