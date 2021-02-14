from decibans import (odds_to_prob, prob_to_odds, factor, posterior_odds,
                      dbans, dbans_to_odds, laplace_smooth, interpret,
                      novelty)


def test_odds_to_prob_a():
    assert odds_to_prob(1) == 0.5


def test_prob_to_odds_a():
    assert prob_to_odds(0.5) == 1.0


def test_factor_a():
    assert factor(0.6, 0.5) == 1.2
