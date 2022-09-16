import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from biasbalancer.balancer import BiasBalancer
# Fairlearn algorithms and utils
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Get the data
data = pd.read_csv(r'biasbalancer\data\processed\german_credit.csv')
data = pd.read_csv(r'biasbalancer\data\processed\catalan-juvenile-recidivism\catalan-juvenile-recidivism-subset.csv')

# one hot encode the data
discrete_cols = ['account_amount', 'credit_history', 'credit_purpose',
            'savings_amount', 'employment_length',
            'other_debtors', 'property', 'installment_plans', 'housing', 'job',
            'telephone', 'foreign_worker']
discrete_cols = ['V4_area_origin', 'V6_province', 'V8_age',
       'V9_age_at_program_end', 'V11_criminal_record', 'V12_n_criminal_record',
       'V13_n_crime_cat', 'V15_main_crime_cat', 'V16_violent_crime',
       'V17_crime_classification', 'V19_committed_crime',
       'V23_territory_of_execution',
       'V24_finished_program', 'V26_finished_measure_grouped',
       'V27_program_duration_cat',
       'V10_date_of_birth_year', 'V10_date_of_birth_month',
       'V22_main_crime_date_year', 'V22_main_crime_date_month',
       'V30_program_start_year', 'V30_program_start_month',
       'V31_program_end_month']
one_hot_encoding = pd.get_dummies(data[discrete_cols])

# add the continuous columns
continous_cols = ['duration', 'credit_amount', 'installment_rate',
             'residence_length', 'age', 'existing_credits', 'dependents']
continous_cols = ['V20_n_juvenile_records', 'V21_n_crime',
                  'V28_days_from_crime_to_program', 'V29_program_duration']
X = data.loc[:, continous_cols].join(one_hot_encoding)
# Y = data['credit_score']
# A = data['sex']
Y = data['V115_RECID2015_recid']
A = data['V1_sex']

# train-test split
X_train, X_test, y_train, y_test, A_train, A_test= train_test_split(
    X,
    Y,
    A,
    test_size=0.3,
    stratify=Y,
    random_state=420)
X_val, X_test, y_val, y_test, A_val, A_test= train_test_split(
    X_test,
    y_test,
    A_test,
    test_size=0.5,
    stratify=y_test,
    random_state=420)

# Train the classifier
# model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=1)
model = MLPClassifier(random_state=1, max_iter=4000, n_iter_no_change=800)
model.fit(X_train, y_train)

# Scores of the classifier
test_scores = model.predict_proba(X_test)[:, 1]
val_scores = model.predict_proba(X_val)[:, 1]

# Predictions (0 or 1) on test set
test_preds = (test_scores >= np.mean(y_train)) * 1
val_preds = (val_scores >= np.mean(y_train)) * 1


# using test data just for now TODO: change to validation data
unaware_model_output = X_val.join(
    pd.DataFrame(
        {'score': val_scores, 'pred': val_preds, 'y': y_val, 'sex': A_val}))
WEIGHT_FP = 0.9
unaware_analysis = BiasBalancer(
    data = unaware_model_output,
    y_name = "y",
    y_hat_name = "pred",
    a_name = "sex",
    r_name = "score",
    w_fp = WEIGHT_FP,
    model_name='Catalan Recidivism')

level1_output_data = unaware_analysis.level_1()
rates, relative_rates, barometer = unaware_analysis.level_2()
_ = unaware_analysis.level_3(method = 'confusion_matrix', **{'cm_print_n': True})

# Fairlearn
postprocess_est = ThresholdOptimizer(
    estimator=model,
    constraints="equalized_odds",
    prefit=True)


# Balanced data set is obtained by sampling the same number of points from the majority class (Y=0)
# as there are points in the minority class (Y=1)
balanced_idx1 = X_train[y_train==1].index
pp_train_idx = balanced_idx1.union(y_train[y_train==0].sample(n=balanced_idx1.size, random_state=1234).index)

X_train_balanced = X_train.loc[pp_train_idx, :]
Y_train_balanced = y_train.loc[pp_train_idx]
A_train_balanced = A_train.loc[pp_train_idx]

postprocess_est.fit(X_train_balanced, Y_train_balanced, sensitive_features=A_train_balanced)

postprocess_preds = postprocess_est.predict(X_val, sensitive_features=A_val)

aware_model_output = X_val.join(
    pd.DataFrame(
        {'score': val_scores, 'pred': postprocess_preds, 'y': y_val, 'sex': A_val}))
WEIGHT_FP = 0.9
aware_analysis = BiasBalancer(
    data = aware_model_output,
    y_name = "y",
    y_hat_name = "pred",
    a_name = "sex",
    r_name = "score",
    w_fp = WEIGHT_FP,
    model_name='Catalan Recidivism')

level1_output_data = aware_analysis.level_1()
rates, relative_rates, barometer = aware_analysis.level_2()
_ = aware_analysis.level_3(method = 'confusion_matrix', **{'cm_print_n': True})
plt.show()
