r"""
Bayesian Statistics

Description:
This is a tool for Bayesian statistical data analysis.

Functionalities:
* Posterior
* Mean, mode and meadian of posterior
* Credible interval
"""


def create_disc_distribution(values, probabilities, normalize=False):
    """
    Create a discrete distribution of a single random variable
    
    Input:
    values: a list of ordered values of a discrete random variable
    probabilities: a list of probabilities of the ordered values
    
    Output:
    dist: a dictionary that represents the probability density distribution
    
    Example:
    import random
    values = range(10)
    probabilities = [random.random for i in xrange(10)]
    sp = sum(probabilities)
    probabilities = [p/sp for p in probabilities]
    dist = create_discrete_distribution(values, probabilities)
    """
    len_p = len(probabilities)
    assert len(values) == len_p, 'The length of values does not match that of probabilities.'
    assert len([i for i in probabilities if i >= 0]) == len_p, 'Probabilities must greater than or equal to 0'
    assert not normalize and sum(probabilities) == 1, 'The sum of probabilities is not 1. Use normalize=True'
    if normalize:
        sp = sum(probabilities)
        probabilities = [p/sp for p in probabilities]
    dist = dict()
    for v, p in values, probabilities:
        dist[v] = p
    return dist
    

def cal_disc_percentile(dist, percentage):
    """
    Calculate the percentile of a single variable discrete distribution
    
    Input:
    dist: a single variable discrete distribution
    percentage: percentage
    
    Output:
    percentile: percentile
    
    Example:
    import random
    values = range(50)
    probabilities = [random.random for i in xrange(50)]
    sp = sum(probabilities)
    probabilities = [p/sp for p in probabilities]
    dist = create_discrete_distribution(values, probabilities)
    percentile = cal_disc_percentile(dist, 0.25)
    """
    total = 0.0
    for i in dist:
        total += dist[i]
        if total > percentage:
            return i


def cal_disc_likelihood(hypo_dist_list, data):
    """
    Calculate the likelihood of an outcome of a single discrete random variable
    
    Input:
    hypo_dist_list: a list of distributions of hypothesises
    data: an outcome of a single discrete random variable 
    
    Output:
    likelihood: likelihood
    
    Example:
    import random
    def generate_dist(num=3):
        dist = []
        for t in xrange(num):
            values = range(10)
            probabilities = [random.random for i in xrange(10)]
            sp = sum(probabilities)
            probabilities = [p/sp for p in probabilities]
            dist.append(create_discrete_distribution(values, probabilities))
        return dist
    data = random.randint(0, 10)
    dist = generate_dist()
    cal_disc_likelihood(dist, data)
    """
    likelihood = []
    for i, dist in enumerate(hypo_dist_list):
        likelihood[i] = dist.get(data, 0)
    return likelihood


def cal_disc_posterior(prior, likelihood):
    """
    Calculate the posterior of an outcome of a single discrete random variable
    
    Input:
    prior: prior density distribution of hypothesises
    likelihood: likelihood
    
    Output:
    posterior: posterior
    
    Example:
    import random
    def generate_dist(num=3):
        dist = []
        for t in xrange(num):
            values = range(10)
            probabilities = [random.random for i in xrange(10)]
            sp = sum(probabilities)
            probabilities = [p/sp for p in probabilities]
            dist.append(create_discrete_distribution(values, probabilities))
        return dist
    data = random.randint(0, 10)
    dist = generate_dist()
    likelihood = cal_disc_likelihood(dist, data)
    prior = generate_dist(num=1)
    cal_disc_posterior(prior, likelihood)
    """
    posterior = []
    num = len(prior)
    assert num == len(likelihood), 'The lengths of prior and likelihood do not match.'
    for i in xrange(num):
        posterior.append(prior[i] * likelihood[i])
    sp = sum(posterior)
    return [p/sp for p in posterior]

def cal_mean_disc_posterior(hypo_val, posterior):
    """
    Calculate the mean of the discrete posterior.
    Assume the hypothesises are a list of values or parameters.
    
    Input:
    hypo_val: the hypothesises, which are a list of values or parameters.
    posterior: posterior
    
    Output:
    mean: the mean of the posterior
    
    Example:
    import random
    hypo_val = range(10)
    posterior = [random.random() for i in xrange(10)]
    posterior = [ i / sum(posterior) for i in posterior]
    print cal_mean_disc_posterior(hypo_val, posterior)
    """
    l = len(hypo_val)
    mean = 0
    assert len(hypo_val) == len(posterior), 'The lengths of hypothesises and posterior do not match.'
    for i in l:
        mean += hypo_val[i] * posterior[i]
    return mean

def cal_mode_disc_posterior(hypo_val, posterior):
    """
    Calculate the mode of the discrete posterior.
    Assume the hypothesises are a list of values or parameters.
    
    Input:
    hypo_val: the hypothesises, which are a list of values or parameters.
    posterior: posterior
    
    Output:
    mode: the mode of the posterior
    
    Example:
    import random
    hypo_val = range(10)
    posterior = [random.random() for i in xrange(10)]
    posterior = [ i / sum(posterior) for i in posterior]
    print cal_mode_disc_posterior(hypo_val, posterior)
    """
    l = len(hypo_val)
    assert len(hypo_val) == len(posterior), 'The lengths of hypothesises and posterior do not match.'
    mode = max_p = 0
    for i in l:
        if posterior[i] > max_p
            max_p = posterior[i]
            mode = hypo_val[i]
    return mode
    
def cal_median_disc_posterior(hypo_val, posterior):
    """
    Calculate the median of the discrete posterior.
    Assume the hypothesises are a list of values or parameters.
    
    Input:
    hypo_val: the hypothesises, which are a list of values or parameters.
    posterior: posterior
    
    Output:
    mid: the median of the posterior
    
    Example:
    import random
    hypo_val = range(10)
    posterior = [random.random() for i in xrange(10)]
    posterior = [ i / sum(posterior) for i in posterior]
    print cal_median_disc_posterior(hypo_val, posterior)
    """
    l = len(hypo_val)
    mid = 0
    assert len(hypo_val) == len(posterior), 'The lengths of hypothesises and posterior do not match.'
    return cal_disc_credible_interval(dict(zip(hypo_val, posterior)), 0.5)

def cal_disc_credible_interval(dist, level):
    """
    Calculate the credible interval of a single variable discrete distribution
    
    Input:
    dist: a single variable discrete distribution
    level: significant level
    
    Output:
    list: the credible interval
    
    Example:
    import random
    values = range(50)
    probabilities = [random.random for i in xrange(50)]
    sp = sum(probabilities)
    probabilities = [p/sp for p in probabilities]
    dist = create_discrete_distribution(values, probabilities)
    print cal_disc_credible_interval(dist, 0.9)
    """
    left = (1 - significant_level) / 2
    right = significant_level + left
    return cal_disc_percentile(dist, left), cal_disc_percentile(dist, right)


def create_beta_distribution_for_binomial(alpha, beta):
    """
    Create a beta distribution for a binomial random variable
    
    Input:
    alpha, beta: when alpha=1 and beta=1, it is uniform from 0 to 1
    
    Output:
    dist: a dictionary with two keys alpha and beta
    
    Example:
    beta = create_beta_distribution_for_binomial(1, 1)
    """
    return {'alpha': alpha, 'beta': beta}


def cal_beta_binomial_posterior(beta_dist, data):
    """
    Calculate the posterior of outcomes of a binomial random variable

    Input:
    beta_dist: beta density distribution
    data: list of outcomes of a binomial random variable.
          1 for success; 0 for failure
    
    Output:
    posterior: posterior
    
    Example:
    import random
    beta_dist = create_beta_distribution_for_binomial(1, 1)
    data = [random.randint(0, 1) for i in xrange(10)]
    post_beta = cal_beta_binomial_posterior(beta_dist, data)
    """
    for i in data:
        if i == 1:
            beta_dist['alpha'] += 1
        else:
            beta_dist['beta'] += 1
    return beta_dist


def cal_mean_beta_binomial_posterior(beta_dist):
    """
    Calculate the mean of a beta distribution for a binomial random variable.
    
    Input:
    beta_dist: beta density distribution

    Output:
    mean: the mean of the beta_dist
    
    Example:
    import random
    beta_dist = create_beta_distribution_for_binomial(1, 1)
    mean = cal_mean_beta_binomial_posterior(beta_dist)
    """
    return float(beta_dist['alpha']) / (beta_dist['alpha'] + beta_dist['beta'])
