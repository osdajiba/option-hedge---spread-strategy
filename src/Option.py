#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from scipy.optimize import newton
from math import exp, log, sqrt
from scipy.stats import norm
import numpy as np
import bisect


def remove_first_row(df):
    return df.drop(df.index[0], inplace=True)

def get_current_price(option_price):
    if isinstance(option_price, dict):
        current_price = next(iter(option_price.values()))
    elif isinstance(option_price, (list, pd.Series)):
        current_price = option_price[0]
    elif isinstance(option_price, pd.DataFrame):
        current_price = option_price.iloc[0]
    elif isinstance(option_price, (float, np.float64, str)):
        current_price = option_price
    else:
        raise TypeError("Unsupported data format for option_price") # 默认值
    return current_price


class Option:
    """
    A class representing an option contract.

    Attributes:
        contract_code (str): The code of the option contract.
        option_price (float): The price of the option.
        underlying_price (float): The price of the underlying asset.
        strike_price (float): The strike price of the option.
        risk_free_rate (float): The risk-free interest rate.
        operation (str): The operation type, either 'buy' or 'sell'.
        volatility (float): The volatility of the underlying asset.
        implied_volatility (float): The implied volatility of the option.
        date (datetime): The date of the option.
        end_date (datetime): The expiration date of the option.
        tau (float): Time to expiration in years.
        Time_to_maturity (float): Time to maturity in days.
        pricing_method (str): The pricing method for the option.
    """

    def __init__(self, contract_code, option_price, underlying_price, strike_price, risk_free_rate,
                 operation="buy", volatility=None, implied_volatility=None,
                 date=None, end_date=None, tau=None, Time_to_maturity=None, pricing_method="Black-Scholes",
                 DELTA_VALUE=None, THETA_VALUE=None, GAMMA_VALUE=None, VEGA_VALUE=None, RHO_VALUE=None):

        self.contract_code = contract_code
        self.option_price = option_price

        self.type = "call" if "-C-" in contract_code else (
            "put" if "-P-" in contract_code else ValueError("Invalid contract_code format"))
        self.underlying_price = float(underlying_price.replace(",", "")) if isinstance(underlying_price,
                                                                                       str) else underlying_price
        self.strike_price = float(strike_price)
        self.risk_free_rate = risk_free_rate
        
        self.operation = operation
        
        self.volatility = np.var(option_price) if volatility is None else volatility

        self.date = date
        self.end_date = end_date
        self.Time_to_maturity = end_date - date if Time_to_maturity is None else Time_to_maturity
        self.tau = self.Time_to_maturity / 365.0 if tau is None else tau

        self.pricing_method = pricing_method

        self.implied_volatility = implied_volatility

        self.DELTA = DELTA_VALUE
        self.THETA = THETA_VALUE
        self.GAMMA = GAMMA_VALUE
        self.VEGA = VEGA_VALUE
        self.RHO = RHO_VALUE

    def next_day(self):
        remove_first_row(self.option_price)
        remove_first_row(self.underlying_price)
        remove_first_row(self.date)
        self.cur_opt_price = get_current_price(self.option_price)
        self.cur_und_price = get_current_price(self.underlying_price)
        
    def calculate_PNL(self):
        """
        Calculate the PNL of the option.

        Returns:
            float: The PNL of the option.
        """
        if self.operation == "buy":
            return self.cur_opt_price - self.strike_price
        elif self.operation == "sell":
            return self.strike_price - self.cur_opt_price
        else:
            raise ValueError("Invalid operation")

    def _args(self):
        """
        Return the order parameters as a dictionary.

        Returns:
            dict: A dictionary containing the order parameters.
        """
        args = {
            'contract_code': self.contract_code,
            'option_type': self.type,
            'option_price': self.option_price,
            'underlying_price': self.underlying_price,
            'strike_price': self.strike_price,
            'risk_free_rate': self.risk_free_rate,
            'operation': self.operation,
            'volatility': self.volatility,
            'implied_volatility': self.implied_volatility,
            'date': self.date,
            'end_date': self.end_date,
            'tau': self.tau,
            'Time_to_maturity': self.Time_to_maturity,
            'pricing_method': self.pricing_method,
            'DELTA_VALUE': self.DELTA,
            'THETA_VALUE': self.THETA,
            'GAMMA_VALUE': self.GAMMA,
            'VEGA_VALUE': self.VEGA,
            'RHO_VALUE': self.RHO
        }
        return args

    def _calculate_black_scholes_price(self, with_implied_volatility=True):
        S = self.underlying_price
        K = self.strike_price
        T = self.tau
        r = self.risk_free_rate
        sigma = self.volatility

        def calculate_iv(sigma):
            d_1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
            d_2 = d_1 - sigma * sqrt(T)
            if self.type == "call":
                return S * norm.cdf(d_1) - K * exp(-r * T) * norm.cdf(d_2) - self.option_price
            elif self.type == "put":
                return K * exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1) - self.option_price
            else:
                raise ValueError("Invalid operation. Must be 'buy' or 'sell' for call option.")

        ivs = np.linspace(0.01, 2.0, 1000)
        prices = [calculate_iv(iv) for iv in ivs]
        idx = bisect.bisect_left(prices, 0)
        self.implied_volatility = ivs[idx] if idx < len(ivs) else ivs[-1]

        if with_implied_volatility:
            sigma = self.implied_volatility

        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if self.type == "call":
            sign = 1 if self.operation == "buy" else -1
            return sign * (S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2))
        else:
            sign = 1 if self.operation == "sell" else -1
            return sign * (K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

    def calculate_price(self):
        # TODO: unfinished pricing model
        if self.pricing_method == "Black-Scholes":
            return self._calculate_black_scholes_price()
        elif self.pricing_method == "BAW":
            return self._calculate_baw_price()
        elif self.pricing_method == "Binomial-Tree":
            return self._calculate_binomial_tree_price()
        elif self.pricing_method == "Monte-Carlo":
            return self._calculate_monte_carlo_price()
        elif self.pricing_method == "Heston":
            return self._calculate_heston_price()
        else:
            raise ValueError(
                "Invalid pricing method. Supported methods are 'Black-Scholes', 'BAW', 'Binomial-Tree', 'Monte-Carlo', and 'Heston'.")

    def _calculate_baw_price(self):
        def calculate_bs_iv():
            bs_price = self._calculate_black_scholes_price()
            return bs_price - self._calculate_black_scholes_price()

        self.implied_volatility = newton(calculate_bs_iv, self.volatility)
        baw_price = self._calculate_black_scholes_price()
        return baw_price

    def _calculate_binomial_tree_price(self):
        # TODO: unfinished pricing model
        raise NotImplementedError("Binomial-Tree model not implemented yet.")

    def _calculate_heston_price(self, kappa=1.5, theta=0.02, sigma=0.3, rho=-0.5, v0=0.02):
        S0 = self.underlying_price
        K = self.strike_price
        T = self.tau
        r = self.risk_free_rate

        n_simulations = 10000
        n_steps = 100
        dt = T / n_steps

        z1 = np.random.normal(size=(n_simulations, n_steps))
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=(n_simulations, n_steps))
        vt = np.zeros((n_simulations, n_steps + 1))
        vt[:, 0] = v0

        for i in range(1, n_steps + 1):
            vt[:, i] = np.maximum(
                vt[:, i - 1] + kappa * (theta - vt[:, i - 1]) * dt + sigma * np.sqrt(vt[:, i - 1] * dt) * z2[:, i - 1],
                0)

        st = np.zeros((n_simulations, n_steps + 1))
        st[:, 0] = S0

        for i in range(1, n_steps + 1):
            st[:, i] = st[:, i - 1] * np.exp((r - 0.5 * vt[:, i]) * dt + np.sqrt(vt[:, i]) * z1[:, i - 1])

        payoff = np.maximum(0, st[:, -1] - K)
        heston_price = np.mean(np.exp(-r * T) * payoff)
        self.implied_volatility = np.std(payoff) / np.mean(payoff)

        return heston_price

    def _calculate_monte_carlo_price(self):
        # TODO: unfinished pricing model
        raise NotImplementedError("Monte-Carlo model not implemented yet.")

    def calculate_greeks(self):
        if self.pricing_method != "Black-Scholes":
            raise NotImplementedError("Only Black-Scholes model is supported for calculating greeks.")

        S = self.underlying_price
        K = self.strike_price
        T = self.tau
        r = self.risk_free_rate
        sigma = self.implied_volatility

        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if self.type == "call":
            self.DELTA = norm.cdf(d1)
            self.THETA = -((S * norm.pdf(d1) * sigma) / (2 * sqrt(T))) - (r * K * exp(-r * T) * norm.cdf(d2))
            self.GAMMA = norm.pdf(d1) / (S * sigma * sqrt(T))
            self.VEGA = S * norm.pdf(d1) * sqrt(T)
            self.RHO = K * T * exp(-r * T) * norm.cdf(d2)
        else:
            self.DELTA = -norm.cdf(-d1)
            self.THETA = -((S * norm.pdf(d1) * sigma) / (2 * sqrt(T))) + (r * K * exp(-r * T) * norm.cdf(-d2))
            self.GAMMA = norm.pdf(d1) / (S * sigma * sqrt(T))
            self.VEGA = S * norm.pdf(d1) * sqrt(T)
            self.RHO = -K * T * exp(-r * T) * norm.cdf(-d2)

        return {'delta': self.DELTA, 'gamma': self.THETA, 'theta': self.GAMMA, 'vega': self.VEGA, 'rho': self.RHO}

    def calculate_iv(self):
        if self.pricing_method != "Black-Scholes":
            raise NotImplementedError("Implied volatility calculation is only supported for Black-Scholes model.")
        self.implied_volatility = self._calculate_black_scholes_price(with_implied_volatility=False)

    def __str__(self):
        return f"Option(contract_code={self.contract_code}, option_price={self.option_price}, underlying_price={self.underlying_price}, strike_price={self.strike_price}, risk_free_rate={self.risk_free_rate}, operation={self.operation}, volatility={self.volatility}, implied_volatility={self.implied_volatility}, date={self.date}, end_date={self.end_date}, tau={self.tau}, Time_to_maturity={self.Time_to_maturity}, pricing_method={self.pricing_method}, DELTA_VALUE={self.DELTA}, THETA_VALUE={self.THETA}, GAMMA_VALUE={self.GAMMA}, VEGA_VALUE={self.VEGA}, RHO_VALUE={self.RHO})"

    def __repr__(self):
        return self.__str__()
