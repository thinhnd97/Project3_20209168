# -*- coding: utf-8 -*-
import os
os.chdir(r'D:\HUST\Project 3')
from process_data import get_dataset, get_apps_processed, get_prev_processed, get_prev_amt_agg, get_prev_refused_appr_agg, get_prev_agg, get_bureau_processed, get_bureau_day_amt_agg, get_bureau_active_agg, get_bureau_bal_agg, get_bureau_agg, get_pos_bal_agg, get_install_agg, get_card_bal_agg, get_apps_all_encoded, get_apps_all_train_test, get_apps_all_with_all_agg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys 
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
# from sklearn.externals import joblib
import joblib


pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 200)
default_dir = r'D:\HUST\Project 3\home-credit-default-risk'


apps, prev, bureau, bureau_bal, pos_bal, install, card_bal = get_dataset()
apps_all = get_apps_all_with_all_agg(apps, prev, bureau, bureau_bal, pos_bal, install, card_bal)
apps_all = get_apps_all_encoded(apps_all)
apps_all_train, apps_all_test = get_apps_all_train_test(apps_all)
del(bureau_bal)
del(card_bal)
del(bureau)
del(install)
del(pos_bal)
del(prev)


# Phân chia tập train, validate và test
from sklearn.model_selection import train_test_split
Z = apps_all_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
Y = apps_all_train['TARGET']
Z_DEV, Z_TEST, Y_DEV, Y_TEST = train_test_split(Z, Y, test_size = 0.1, random_state = 0)
Z_TRAIN, Z_VAL, Y_TRAIN, Y_VAL = train_test_split(Z_DEV, Y_DEV, test_size = 0.2, random_state = 0)
# Lưu lại data tập test
writer = pd.ExcelWriter(f'D:\HUST\Project 3\Testing data\Testing_Data.xlsx', engine='xlsxwriter')
Z_TEST.to_excel(writer, sheet_name='Z', index=False)
Y_TEST.to_excel(writer, sheet_name='Y', index=False)
writer.save()
writer.close()


# Train model
# https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
# https://lightgbm.readthedocs.io/en/latest/Parameters.html
clf = LGBMClassifier(
            n_estimators=4000, # Number of boosted trees to fit (default=100)
            learning_rate=0.01, # Boosting learning rate (default=0.1)
            max_depth = 11, # Maximum tree depth for base learners, <=0 means no limit (default=-1)
            num_leaves=58, # Maximum tree leaves for base learners (default=31)
            colsample_bytree=0.613, # Subsample ratio of columns when constructing each tree (default=1.)
            subsample=0.708, # Subsample ratio of the training instance (default=1.)
            max_bin=407, # max number of bins that feature values will be bucketed in (default=255)
            reg_alpha=3.564,
            reg_lambda=4.930,
            min_child_weight= 6,
            min_child_samples=165,
            silent=-1,
            verbose=-1,
            )
clf.fit(Z_TRAIN, Y_TRAIN, eval_set=[(Z_TRAIN, Y_TRAIN), (Z_VAL, Y_VAL)], eval_metric= 'auc', verbose= 200, early_stopping_rounds= 200)


# Lưu kết quả đánh giá trên tập test
#predict = clf.predict_proba(Z_TRAIN, num_iteration=clf.best_iteration_)[:, 1]
predict = clf.predict_proba(Z_TEST, num_iteration=clf.best_iteration_)[:, 1]
score = predict*1000
score = score.astype(int)
#result = pd.DataFrame({'LABEL':Y_TRAIN,'SCORE':score})
result = pd.DataFrame({'LABEL':Y_TEST,'SCORE':score})


bins = [min(score) - 100]
for i in range(1, 11):
    if i < 10:
        a = int(result[['SCORE']].quantile(0.1 * i)[0])
    else:
        a = int(result[['SCORE']].quantile(0.1 * i)[0]) + 100
    bins.append(a)
print(bins)
bins = [-99,12,17,24,32,43,57,79,115,191,931]
result['score_range'] = pd.cut(result['SCORE'], bins = bins).astype(str)
default_dir = r'D:\HUST\Project 3'
result.to_excel(os.path.join(default_dir,'result.xlsx'), index=False)


print("Best iteration: ", clf.best_iteration_)
feature_importances = pd.DataFrame({'FEATURE':Z_TRAIN.columns, 'IMPORTANCE':clf.feature_importances_})
feature_importances = feature_importances.sort_values(by='IMPORTANCE', ascending=False)
feature_importances.to_excel(os.path.join(default_dir,'feature_importance.xlsx'), index=False)


# Save model
joblib.dump(clf, r'D:\HUST\Project 3\lightgbm_final.p')