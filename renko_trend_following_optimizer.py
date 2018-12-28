from hyperopt import hp, tpe, fmin, Trials
import numpy as np
import pandas as pd
import datetime
import pyrenko
import scipy.optimize as opt
from scipy.stats import iqr

from catalyst import run_algorithm
from catalyst.api import (record, symbol, order_target, order_target_percent, get_datetime)

period_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

start_date = datetime.datetime.strptime("1-6-2018", "%d-%m-%Y")
end_date = datetime.datetime.strptime("31-10-2018", "%d-%m-%Y")
total_days = (end_date - start_date).days + 1

folds_start_date = [start_date] + [start_date + datetime.timedelta(days = round(total_days * x)) for x in np.cumsum(period_weights)[:-1]]
folds_end_date = [start_date + datetime.timedelta(days = round(total_days * x) - 1) for x in np.cumsum(period_weights)]

hyper_params_space = {'n_history': hp.quniform('n_history', 150, 1000, 1),
'tf': hp.quniform('tf', 10, 100, 1),
'diff_lag': hp.quniform('diff_lag', 1, 100, 1),
'part_cover_ratio': hp.uniform('part_cover_ratio', 0.0, 1.0)}

def weighted_mean(values):
    return np.average(values, weights = list(range(1, len(values) + 1)))
    
def score_func(params):
	# Function for Renko brick optimization
    def evaluate_renko(brick, history, column_name):
        renko_obj = pyrenko.renko()
        renko_obj.set_brick_size(brick_size = brick, auto = False)
        renko_obj.build_history(prices = history)
        return renko_obj.evaluate()[column_name]

    def initialize(context):
        context.asset = symbol('eth_btc')

        context.leverage = 1.0              		          			# 1.0 - no leverage
        context.n_history = int(params['n_history'])                    # Number of lookback bars for modelling
        context.tf = str(int(params['tf'])) + 'T'                      	# How many minutes in a timeframe
        context.diff_lag = int(params['diff_lag'])                      # Lag of differences to get returns
        context.model = pyrenko.renko()     		   		  			# Renko object
        context.part_cover_ratio = float(params['part_cover_ratio']) 	# Partially cover position ratio
        context.last_brick_size = 0.0       		          			# Last optimal brick size (just for storing)
    
        context.set_benchmark(context.asset)
        context.set_commission(maker = 0.001, taker = 0.002)
        context.set_slippage(slippage = 0.0005)

    def handle_data(context, data):
        current_time = get_datetime().time()

        # When model is empty
        if len(context.model.get_renko_prices()) == 0:
            context.model = pyrenko.renko()
            history = data.history(context.asset,
                'price',
                 bar_count = context.n_history, 
                frequency = context.tf
                )

            # Get daily absolute returns
            diffs = history.diff(context.diff_lag).abs()
            diffs = diffs[~np.isnan(diffs)]
            # Calculate IQR of daily returns
            iqr_diffs = np.percentile(diffs, [25, 75])

            # Find the optimal brick size
            opt_bs = opt.fminbound(lambda x: -evaluate_renko(brick = x,
            	history = history, column_name = 'score'),
            iqr_diffs[0], iqr_diffs[1], disp=0)

            # Build the model
            context.last_brick_size = opt_bs
            context.model.set_brick_size(brick_size = opt_bs, auto = False)
            context.model.build_history(prices = history)

            # Open a position
            order_target_percent(context.asset, context.leverage * context.model.get_renko_directions()[-1])

        else:
            last_price = data.history(context.asset,
            	'price',
            	bar_count = 1,
            	frequency = '1440T',
            	)

            # Just for output and debug
            prev = context.model.get_renko_prices()[-1]
            prev_dir = context.model.get_renko_directions()[-1]
            num_created_bars = context.model.do_next(last_price)

            # If the last price moves in the backward direction we should rebuild the model
            if np.sign(context.portfolio.positions[context.asset].amount * context.model.get_renko_directions()[-1]) == -1:
                order_target_percent(context.asset, 0.0)
                context.model = pyrenko.renko()
            # or we cover the part of the position
            elif context.part_cover_ratio > 0.0 and num_created_bars != 0:
                order_target(context.asset, context.portfolio.positions[context.asset].amount * (1.0 - context.part_cover_ratio))
    
    def analyze(context, perf):
    	pass

    # Run alfo and get the performance
    perf = run_algorithm(
    	capital_base = 1000000,
    	data_frequency = 'daily',
    	initialize = initialize,
    	handle_data = handle_data,
    	analyze = analyze,
    	exchange_name = 'bitfinex',
    	quote_currency = 'btc',
    	start = pd.to_datetime(params['start'], utc = True),
    	end = pd.to_datetime(params['end'], utc = True))

	# Invert the metric
    if pd.isnull(perf.sortino[-1]):
    	return 0.0
    else:
    	return (-1.0) * perf.sortino[-1]

def objective(hyper_params):
    print(hyper_params)

    # Calculate metric for each fold
    metric_folds = [0.0] * (len(folds_start_date))
    for p in range(len(folds_start_date)):
        hyper_params['start'] = folds_start_date[p]
        hyper_params['end'] = folds_end_date[p]
		
        metric_folds[p] = score_func(hyper_params)
        print('Fold #' + str(p) +' metric value: ' + str(metric_folds[p]))
    
    result = 0.0
    if np.max(metric_folds) >= 0.0:
        result = np.max(metric_folds)
    else:
        result = weighted_mean(metric_folds)

    print('Objective function value: ' + str(result))
    return result

tpe_trials = Trials()
opt_params = fmin(fn = objective,
            space = hyper_params_space,
            algo = tpe.suggest, 
            max_evals = 300,
            trials = tpe_trials, 
            rstate = np.random.RandomState(100))

tpe_results = pd.DataFrame({'score': [x['loss'] for x in tpe_trials.results], 
                            'n_history': tpe_trials.idxs_vals[1]['n_history'],
                            'tf': tpe_trials.idxs_vals[1]['tf'],
                            'diff_lag': tpe_trials.idxs_vals[1]['diff_lag'],
                            'part_cover_ratio': tpe_trials.idxs_vals[1]['part_cover_ratio']})
tpe_results.sort_values(by = ['score'], inplace = True)

print(tpe_results.head(10))
print(opt_params)