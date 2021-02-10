from numpy import log10
from pandas import DataFrame


def prob_to_odds(p):
    """Convert probability to odds (on to one)."""
    odds = p / (1 - p)
    return odds


def odds_to_prob(odds):
    """Convert odds (on to one) to probability."""
    p = 1 - (1 / (1 + odds))
    return p


def factor(p_obs_true, p_obs_false):
    """Calculate the Bayes factor, given the probabilities of making the
    observiation if the theory is true and false."""
    K = p_obs_true / p_obs_false
    return K


def posterior_odds(prior_odds, p_obs_true, p_obs_false):
    """Calculate posterior odds (on to one), given prior odds (on to one) and
    the probabilities of making the observiation if the theory is true and
    false.
    """
    K = factor(p_obs_true, p_obs_false)
    post_odds = prior_odds * K
    return K, post_odds


def dbans(odds):
    """Express the odds (on to one) as decibans (10log_{10}(odds))"""
    dbn = 10 * log10(odds)
    return dbn


def dbans_to_odds(dbans):
    """Convert decibans to odds (on to one)."""
    odds = 10 ** (dbans / 10)
    return odds


def laplace_smooth(alpha, occurences, trials, outcomes):
    """Calculate a Laplace smoothed probability.

    alpha: smoothing parameter, e.g. 0.1, 0.5 (Jeffreys), or 1 (Laplace).
    occurences: number of trials with the outcome of interest.
    trials: the number of trials in which the occurences were observed.
    """
    p_star = (occurences + alpha) / (trials + alpha * outcomes)
    return p_star


def interpret(factor):
    """Interpret the strength of evidence for the supplied factor."""
    if 0 < factor < 1:
        return 'Evidence supports the alternative model.'
    elif factor == 1:
        return 'The evidence supports both models equally.'
    elif 1 < factor <= 3.2:
        return 'Evidence is barely worth mentioning.'
    elif 3.2 < factor <= 10:
        return 'Evidence is substantial.'
    elif 10 < factor <= 100:
        return 'Evidence is strong.'
    elif factor > 100:
        return 'Evidence is decisive.'
    else:
        raise ValueError('The factor should be a positive real number.')


def novelty(data_list):
    """Find the first occurence of each novel item in the data list.

    data_list is assumed to be in order of occurence.
    Returns a pandas dataframe with columns:
        item: the item from the supplied data_list.
        first_index: the index of the first occurence of the present item.
        frequency: the frequency of the item in data_list.
        running_total: the number of novel items present in data_list up to and
                       including the first appearance of the present item.
    """
    items = set(data_list)
    novel_indices = [data_list.index(i) for i in items]
    frequency = [data_list.count(i) for i in items]
    running_total = [len(set(data_list[0:n+1])) for n in novel_indices]
    df = DataFrame({'item': list(items), 'first_index': novel_indices,
                    'frequency': frequency, 'running_total': running_total})
    return df
