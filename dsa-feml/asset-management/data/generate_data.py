import pandas as pd
import numpy as np

from data.helpers import calculate_spot, simulation_spot, d1, d2, call
from data.constants import s_0, vol, r, T, n, k


def generate_data():
    simulation = pd.DataFrame()
    call_prices = []
    maturity = []

    simulation['Sim_1'] = simulation_spot(s_0, r, n, T, vol)
    simulation.index = np.round(np.arange(0, 0.5 + (0.5 / 1000), 0.5 / 1000), 4)

    for (a, b) in zip(simulation['Sim_1'], simulation.index):
        if b != T:
            d1_ = d1(a, k, r, b, T, vol)
            d2_ = d2(a, k, r, b, T, vol)

            call_prices.append(call(d1_, d2_, k, r, T, b, a))
            maturity.append((T - b))

        else:
            call_prices.append(max(a - k, 0))
            maturity.append(0)

    simulation.insert(0, 'Sim_1_Call', call_prices)
    simulation.index = pd.date_range(start='01/01/2018', end='06/01/2018', periods=1001)

    simulation['Maturity'] = maturity
    simulation['Strike'] = k
    simulation['Risk_Free'] = r
    simulation['Volatility'] = vol

    simulation.to_csv('data/asset_data.csv')
