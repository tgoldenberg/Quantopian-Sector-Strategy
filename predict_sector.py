import numpy as np
import talib
import pandas as pd
import blaze
import matplotlib.pyplot as plt
import datetime
from quantopian.pipeline import Pipeline, CustomFilter
from quantopian.pipeline.filters import StaticAssets
from quantopian.research import run_pipeline
from quantopian.pipeline.factors import Returns, CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.interactive.data.quandl import rateinf_inflation_usa # Inflation rate (1yr)
from quantopian.interactive.data.quandl import fred_gnp # GNP 90d
from quantopian.interactive.data.quandl import fred_gdp # GDP 1yr
from quantopian.interactive.data.quandl import fred_icsa # Unemployment Claims 30d
from quantopian.interactive.data.quandl import adp_empl_sec # Total Jobs 30d
from quantopian.pipeline.data.quandl import fred_unrate # unemployment rate (test)

# convert all economic data to Pandas.DataFrame objects
jobs = blaze.compute(adp_empl_sec).sort_values("asof_date")
gnp = blaze.compute(fred_gnp).sort_values("asof_date")
gdp = blaze.compute(fred_gdp).sort_values("asof_date")
inflation = blaze.compute(rateinf_inflation_usa).sort_values("asof_date")
unemployment = blaze.compute(fred_icsa).sort_values("asof_date")

working_ids = [ 19658, 19656 ]

all_ids = [
    19658, # XLK - technology   0
    19656, # XLF - financials   1
    19654, # XLB - materials    2
    19657, # XLI - industrials  3
    26670, # VOX - telecom      4
    26669, # VNQ - real estate  5
    19661, # XLV - health care  6
    19660, # XLU - utilities    7
    19662, # XLY - consumer discretionary 8
    19659, # XLP - consumer staples 9
    19655, # XLE - energy 10
    8554,  # SPY 11
]

features = {
    "19658": { # XLK - Technology
        "macro": [
            0, # jobs change
            3, # inflation change
        ],
        "sector": [
            11, # SPY
            8, # XLY
            6, # XLV
        ],
    },
    "19656": { # XLF - Financials
        "macro": [
            3, # inflation
            0, # jobs change
            1, # GNP
        ],
        "sector": [
            11, # SPY
            3, # XLI
        ]
    }
}

LOOKBACK = 5


#### Helper Functions
def find_most_recent(date, frame = gnp):
    for i in range(len(frame)):
        asof_date = frame.iloc[i].asof_date
        date = date.replace(tzinfo=None)
        if asof_date > date:
            latest = frame.iloc[i-1].value
            prev = frame.iloc[i-2].value
            pct_change = latest / prev - 1
            return pct_change

def find_most_recent_jobs(date):
    for i in range(len(jobs)):
        asof_date = jobs.iloc[i].asof_date
        date = date.replace(tzinfo=None)
        if asof_date > date:
            latest = jobs.iloc[i-1].total_private
            prev = jobs.iloc[i-2].total_private
            pct_change = latest / prev - 1
            return pct_change

def compute_returns(id, assets, close):
    idx = np.where(assets == id)
    prices = close.T[idx][0]
    start, end = prices[0], prices[24]
    result = end / start - 1
    return result

def compute_sector_returns(assets, close):
    result = [ ]
    for i in range(len(all_ids)):
        id = all_ids[i]
        monthly_return = compute_returns(id, assets, close)
        result.append(monthly_return)
    return result

#### Custom Factor
class TrainingData(CustomFactor):
    inputs = [USEquityPricing.close.latest]
    outputs = [
        "is_recommended",
        "predicted_return",
    ]
    window_length = LOOKBACK + 25
    def compute(self, today, assets, out, close):
        is_recommended = np.zeros(out.shape) # initialize recommendations with zeros
        all_monthly_returns = compute_sector_returns(assets, close) # compute all sector 1M returns

        # compute macroeconomic data
        jobs_change = find_most_recent_jobs(today)
        gnp_change = find_most_recent(today, gnp)
        gdp_change = find_most_recent(today, gdp)
        inflation_change = find_most_recent(today, inflation)
        unemployment_change = find_most_recent(today, unemployment)
        macroeconomic = [ jobs_change, gnp_change, gdp_change, inflation_change, unemployment_change ]

        # iterate over all assets
        L = len(assets)
        for i in range(L):
            # determine lookback series of returns for asset
            asset = assets[i]
            is_recommended = 0
            predicted_return = 0
            if asset in working_ids:
                prices = close.T[i]
                series = [ ]
                for j in range(0, len(prices) - 25):
                    start = prices[j]
                    end = prices[j + 25]
                    r = start / end - 1
                    series = [r] + series
                X = []
                monthly_return = series[-1]
                X += series[:-1] # add time series
                macro_indices = features[str(asset)]["macro"]
                sector_indices = features[str(asset)]["sector"]
                for j in range(len(macro_indices)):
                    idx = macro_indices[j]
                    X.append(macroeconomic[idx])
                for j in range(len(sector_indices)):
                    idx = sector_indices[j]
                    X.append(all_monthly_returns[idx])
                X.append(monthly_return)
                X = np.array(X)

            # use asset id to determine which features (if/else)
            # run prediction
            # save value in is_recommended[i] = 1 || 0
