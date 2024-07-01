import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials,anneal
import timeit
from scipy.stats import randint
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as rmse
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import make_scorer



if __name__ == "__main__":
    path = 'data.xlsx'


    # pandas读入
    data = pd.read_excel(path)
    # print(data)
    x = data[['h1', 'h2', 'h3','h4','h5']]
    y = data['q']

    #默认参数模型
    clf=RandomForestRegressor()
    clf.fit(x,y)
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=420)
    score = cross_val_score(clf, x, y, cv=cv, scoring='r2').mean()
    print(score)
    print(clf.feature_importances_)

    # 贝叶斯优化
    def hyperopt_train_test(params):
        clf = RandomForestRegressor(**params)
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        score = cross_val_score(clf, x, y, cv=cv, scoring='r2').mean()
        return score


    # 定义参数空间
    space_svm = {'n_estimators': hp.randint('n_estimators', 1, 200)
        , 'max_features': hp.randint('max_features', 1, 7)
        , 'max_depth': hp.randint('max_depth', 1, 20)
        , 'min_samples_split': hp.randint('min_samples_split', 2, 20)
        , 'min_samples_leaf': hp.randint('min_samples_leaf', 1, 20)
                 }

    start = timeit.default_timer()
    # 定义最小化目标函数
    def fn_rf(params):
        mape = hyperopt_train_test(params)
        return {'loss': -mape, 'status': STATUS_OK}
    trials = Trials()
    best = fmin(fn_rf, space_svm, algo=tpe.suggest, max_evals=2000, trials=trials)

    end = timeit.default_timer()

    print("Best；{}".format(best))
    print('Running time:%.5fs' % (end - start))
