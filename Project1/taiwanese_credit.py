#%% inits
from biasbalancer.balancer import BiasBalancer
import pandas as pd


#%% getting data

df_x = pd.read_csv(r'..\biasbalancer\data\processed\taiwanese_credit.csv')
df_nn = pd.read_csv(r'..\biasbalancer\data\predictions\taiwanese_nn_pred.csv')
df_lr = pd.read_csv(r'..\biasbalancer\data\predictions\taiwanese_log_reg.csv')

# %% initialise biasbalancer looking at gender

false_positive_weight = 5 / (5 + 1) # gotten from https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
nn_analysis = BiasBalancer(
    data = df_nn,
    y_name='default_next_month',
    y_hat_name='nn_pred',
    a_name='sex',
    r_name='nn_prob',
    w_fp=false_positive_weight,
    model_name='Taiwanese credet NN'
)
lr_analysis = BiasBalancer(
    data = df_lr,
    y_name='default_next_month',
    y_hat_name='log_reg_pred',
    a_name='sex',
    r_name='log_reg_prob',
    w_fp=false_positive_weight,
    model_name='Taiwanese credet LR'
)

#%% level one analysis



