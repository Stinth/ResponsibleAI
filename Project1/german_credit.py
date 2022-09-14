#%% inits
from biasbalancer.balancer import BiasBalancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% getting and cleaning the data
data = pd.read_csv(r'..\biasbalancer\data\processed\german_credit.csv')

# creating one hot encoder
def one_hot_encode(df: pd.DataFrame, columns=list()):
    if not columns:
        columns = df.columns
    temp = pd.DataFrame()
    for column in columns:
        unique = np.unique(df[column])
        if len(unique) > 2:
            for u in unique:
                temp[column + '_' + u] = np.array(df[column] == u, dtype=int)
        elif len(unique) == 2:
            u = unique[0]
            temp[column + '_' + str(u)] = np.array(df[column] == u, dtype=int)
    return temp

# categorize data columns
discreet = ['account_amount', 'credit_history', 'credit_purpose',
            'savings_amount', 'employment_length',
            'other_debtors', 'property', 'installment_plans', 'housing', 'job',
            'telephone', 'foreign_worker']
continuos = ['duration', 'credit_amount', 'installment_rate',
             'residence_length', 'age', 'existing_credits', 'dependents']
sensitive = ['person_id', 'sex', 'personal_status', 'credit_score']

# one hot encoding the dataset
data_ohe = one_hot_encode(data, columns=discreet)

for c in continuos:
    data_ohe[c] = data[c]

# save for later
non_sensitive_columns = data_ohe.columns

# add sex so seperated correctly
data_ohe['sex'] = data['sex']

#%% creating and training the classifier

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# define data
X = data_ohe
y = data['credit_score']

# seed for consistensy
np.random.seed(42)

# split into test and par
X_par, X_test, y_par, y_test = train_test_split(X, y, stratify=y, random_state=1)

# split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=1)

# save and remove sensitive data
sensitive_train = X_train['sex']
sensitive_val = X_val['sex']
sensitive_test = X_test['sex']

X_train = X_train[non_sensitive_columns]
X_val = X_val[non_sensitive_columns]
X_test = X_test[non_sensitive_columns]

# create and train classifier
clf = MLPClassifier(random_state=1, max_iter=4000, n_iter_no_change=100).fit(X_train, y_train)


#%% optimize the classifier with profit and fairness constraint

# resolution for analysis
resolution = 101
independance_tolerance = 0.2

# calculate the profit loss
def calc_loss(y, pred_probs, tau):
    y = y.to_numpy()
    false_negatives = (y == 1) & (pred_probs < tau)
    false_positives = (y == 0) & (pred_probs >= tau)

    return false_negatives.sum() * 1 + false_positives.sum() * 2

# inits for loss analysis
tau = np.linspace(0, 1, resolution, endpoint=True)
female_mask = sensitive_val.to_numpy() == 'Female'
percent_female = np.mean(female_mask)

# calculate loss'
loss = list(map(lambda tau: calc_loss(y_val, clf.predict_proba(X_val)[:,1], tau), tau))
loss_male = list(map(lambda tau: calc_loss(y_val[~female_mask], clf.predict_proba(X_val)[:,1][~female_mask], tau), tau))
loss_female = list(map(lambda tau: calc_loss(y_val[female_mask], clf.predict_proba(X_val)[:,1][female_mask], tau), tau))

# save taus
best_profit_tau = tau[np.argmin(loss)]
best_profit_tau_male = tau[np.argmin(loss_male)]
best_profit_tau_female = tau[np.argmin(loss_female)]

# calculate probability of getting loan
p_r_male = lambda tau: np.mean(clf.predict_proba(X_val)[:,1][~female_mask] >= tau)
p_r_female = lambda tau: np.mean(clf.predict_proba(X_val)[:,1][female_mask] >= tau)

# get probs of getting a loan
prob_loan_male = list(map(p_r_male, tau))
prob_loan_female = list(map(p_r_female, tau))


# loss, male_i, female_i
independance_best_tau = (float('inf'),0,0)
# get zone where independance holds
independance_zone = np.zeros((resolution, resolution))
for male_i, prob_male in enumerate(prob_loan_male):
    for female_i, prob_female in enumerate(prob_loan_female):
        is_independant = max(prob_male/prob_female, prob_female/prob_male) - 1 < independance_tolerance
        independance_zone[female_i, male_i] = is_independant
        if is_independant:
            loss_male = calc_loss(y_val[~female_mask], clf.predict_proba(X_val)[:,1][~female_mask], tau[male_i])
            loss_female = calc_loss(y_val[female_mask], clf.predict_proba(X_val)[:,1][female_mask], tau[male_i])
            weited_loss = percent_female * loss_female + (1-percent_female) * loss_male
            if weited_loss < independance_best_tau[0]:
                independance_best_tau = (weited_loss, tau[male_i], tau[female_i])

# show zone
# x_axis: tau*resolution for males
# y_axis: tau*resolution for females
# 1 if independance is met
plt.matshow(independance_zone)
plt.show()

# print threshholds
print(f"Tau optimized for profit (male/female):\n{best_profit_tau_male:.2}/\
{best_profit_tau_female:.2}\nIndipendant constaint:\n\
{independance_best_tau[1]:.2}/{independance_best_tau[2]:.2}")




#%% fairness analysis

# choose which taus to use for analysis
tau_male = best_profit_tau_male
tau_female = best_profit_tau_female

# choose bias balancer weight
false_positive_weight = 0.1

# get test probs
nn_pred_prob = clf.predict_proba(X_test)[:,1]

# classify using tau and sensitive group
nn_pred = [prob >= tau_male if gender == 'Male' else prob >= tau_female
           for prob, gender in zip(nn_pred_prob, sensitive_test)]

# save data to dataframe for analysis
df_nn = pd.DataFrame()
df_nn['credit_score'] = y_test
df_nn['nn_pred'] = nn_pred
df_nn['sex'] = sensitive_test.to_numpy()
df_nn['nn_prob'] = nn_pred_prob


# create analysis class
analysis = BiasBalancer(
    data = df_nn,
    y_name='credit_score',
    y_hat_name='nn_pred',
    a_name='sex',
    r_name='nn_prob',
    w_fp=false_positive_weight,
    model_name='German credet NN'
)

