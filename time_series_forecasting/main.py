import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def main():
    df = pd.read_csv("./data/PJME_hourly.csv")
    df.set_index('Datetime', inplace=True)
    df.index = pd.to_datetime(df.index)
    
    color_pal= sns.color_palette()
    plt.style.use('fivethirtyeight')
    
    # BASIC TOTAL DATA PLOT
    # df.plot(style='.', figsize=(13, 4), color=color_pal[0], title="PJME Energy Use in MW")
    
    df = create_features(df)
    train = df.loc[df.index < '01-01-2015']
    test = df.loc[df.index >= '01-01-2015']
    FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year',
       'dayofyear']
    TARGET = ['PJME_MW']
    
    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]
    
    
    # TRAIN-TEST SPLIT VISUAL
    # fig, ax = plt.subplots(figsize=(13,4))
    # train.plot(ax=ax, label='Training Set')
    # test.plot(ax=ax,label='Testing Set')
    # ax.axvline('01-01-2015', color='black', ls='--')
    # ax.legend(['Training Set', 'Testing Set'])
    # plt.show()

    # WEEK DATA VISUAL
    # df.loc[(df.index > '01-01-2010') & (df.index < '01-08-2010')].plot(figsize=(13,4), title='Week of Data')
    # plt.show()
   
    # DAY DATA VISUAL
    # fig, ax = plt.subplots(figsize=(10,8))
    # sns.boxplot(data=df, x='hour', y='PJME_MW')
    # ax.set_title('MW by Hour')
    # plt.show()

    # CREATE MODEL
    reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, learning_rate=0.01)
    reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
    
    fi = pd.DataFrame(data=reg.feature_importances_, index=reg.feature_names_in_, columns=['importance'])
    fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
    plt.show()
    
    # FORECAST ON TEST
    test['prediction'] = reg.predict(X_test)
    df = df.merge(test['prediction'], how='left', left_index=True, right_index=True)
    
    ax = df['PJME_MW'].plot(figsize=(13,4))
    df['prediction'].plot(ax=ax, style='.')
    ax.set_title('Raw Data and Prediction')
    plt.show()
    
    ax = df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['PJME_MW'].plot(figsize=(13,4), title='Week of Data')
    df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['prediction'].plot(style='.')
    plt.legend(['True Data', 'Prediction'])
    plt.show()

# FEATURE CREATION
def create_features(df):
    """
    Creates time series features based on time series index
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df
    

if __name__ == "__main__":
    main()