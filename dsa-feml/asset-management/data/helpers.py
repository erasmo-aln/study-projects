import numpy as np
from scipy import stats

from data import constants


def calculate_spot(prev, sigma, r, step, random):
    spot = prev + (sigma * prev * random) + (r * prev * step)

    return spot


def simulation_spot(s0, r, steps, maturity, vol):
    delta_t = constants.T / steps
    time = np.round(np.arange(0, maturity + delta_t, delta_t), 4)
    prices = [s0]
    normal_dist = np.random.normal(0, np.sqrt(delta_t), 10000)

    for step in range(steps):
        prices.append(calculate_spot(prices[-1], vol, r, delta_t, normal_dist[step]))

    return prices


def d1(s, k, r, t, T, vol):
    if T != t:
        nomin = np.log(s/k) + ((r + 0.5*(vol**2)) * (T - t))
        denom = vol * np.sqrt(T - t)

        return nomin / denom

    return None


def d2(s, k, r, t, T, vol):
    if T != t:
        nomin = np.log(s/k) + ((r - 0.5 * (vol**2)) * (T - t))
        denom = vol * np.sqrt(T - t)

        return nomin / denom

    return None


def call(d1, d2, k, r, T, t, s):
    elem_1 = s * stats.norm.cdf(d1)
    elem_2 = k * np.exp(-r * (T - t))
    elem_3 = stats.norm.cdf(d2)

    result = elem_1 - (elem_2 * elem_3)

    return result
