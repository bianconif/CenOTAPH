import collections
import itertools

def prime_factors(n):
    i = 2
    while i * i <= n:
        if n % i == 0:
            n /= i
            yield i
        else:
            i += 1

    if n > 1:
        yield n


def _prod(iterable):
    result = 1
    for i in iterable:
        result *= i
    return result


def get_divisors(n):
    """Divisors of n
    
    Parameters
    ----------
    n : int
        The number of which we want to compute the divisors
        
    Credits
    -------
    Sourced from: https://alexwlchan.net/2019/07/finding-divisors-with-python/
    """
    pf = prime_factors(n)

    pf_with_multiplicity = collections.Counter(pf)

    powers = [
        [factor ** i for i in range(count + 1)]
        for factor, count in pf_with_multiplicity.items()
    ]

    for prime_power_combo in itertools.product(*powers):
        yield _prod(prime_power_combo)
