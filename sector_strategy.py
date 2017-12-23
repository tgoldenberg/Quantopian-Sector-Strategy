'''
    Training Strategy:
        * run regression analysis (with lag) with all 16 variables
        * sort by Coefficient and P-value
        * choose best 5 values as features to begin training
        * run model with sklearn and tensorflow and get parameters
    Strategy:
        1st day of every month, do the following:
        * close out all positions
        * get all necessary data
            - retrieve 30d EMA (exponential moving average) of the following 11 sector ETFs:
                ($XLK, $XLF, $XLE, $XLB, $XLV, $XLY, $XLP, $XLI, $VNQ, $VOX, $XLU)
            - also get 30d EMA of $SPY
            - get 30d change in total private jobs (ADP National Employment Report)
            - get 90d change in GNP
            - get 365d change in US inflation
            - get 365d change in US GDP
            - get 30d change in unemployment claims
        * run predictive model for each sector ETF
        * rank positive ones for long, rank negative ones for shorting
        * place trades
'''

def sector_strategy():
    return
