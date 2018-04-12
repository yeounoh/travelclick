import matplotlib.pyplot as plt, mpld3
import statsmodels.api as sm
import seaborn as sb
sb.set_style('darkgrid')
import numpy as np
import pandas as pd
import pickle


# Load the demand data
#data = pd.read_csv('sample.dsv',sep='|')
data = pd.read_csv('TC_Market_DMDData.dsv',sep='|')
data = data.sort_values(by=['PROPERTY_IDENTIFIER','STAY_DATE'])
data['PROPERTY_IDENTIFIER'] = pd.to_numeric(data['PROPERTY_IDENTIFIER'])
data['STAY_DATE'] = pd.to_datetime(data['STAY_DATE'])

"""
# Weekly historical (occupancy) data from hotels in Atlanta, GA
ranks = ['', 'GREEN', 'BLACK', 'RED']
ranks = ['']
for r in ranks: 
    atlData = data[data['MARKET'] == 'Atlanta']
    if r != '': atlData = atlData[atlData['MARKET_RANK'] == r]

 tlData   atlData = atlData.set_index(['STAY_DATE'])
    atlData = atlData.groupby(atlData.index)['OCCUPANCY'].mean() # avg occupancy over market
    atlData = atlData.reindex(pd.date_range(atlData.index[0], atlData.index[-1]))
    atlData.interpolate(inplace=True)
    atlData = atlData.resample('W').mean()
    #atlData = atlData.resample('D').mean()

    if r == '': pickle.dump(atlData.ix['2014-01-01':'2015-01-01'], open('atlData.p','wb'))
    else: pickle.dump(atlData.ix['2014-01-01':'2015-01-01'], open('atlData'+r+'.p','wb'))
    if r == '': pickle.dump(atlData.ix['2015-01-01':'2016-01-01'], open('atlData2.p','wb'))
    else: pickle.dump(atlData.ix['2015-01-01':'2016-01-01'], open('atlData'+r+'2.p','wb'))
"""
# With avg_lead_time
ranks = ['']
for r in ranks: 
    atlData = data[data['MARKET'] == 'Atlanta']
    if r != '': atlData = atlData[atlData['MARKET_RANK'] == r]

    atlDataIndexed = atlData.set_index(['STAY_DATE'])

    for i in range(101):
	    tmp = atlData[atlData['AVG_LEAD_TIME'] >= i][['STAY_DATE','OCCUPANCY']]
	    tmp.interpolate(inplace=True)	
	    tmp = tmp.groupby('STAY_DATE')['OCCUPANCY'].mean()
	    atlDataIndexed['%s'%i] = tmp

    atlData = atlDataIndexed
    atlData = atlData.sort_index(axis=0)
    
    cols = [str(i) for i in range(101)]
    cols.append('OCCUPANCY')
    atlData = atlData.groupby(atlData.index)[cols].mean()
    atlData = atlData.reindex(pd.date_range(atlData.index[0], atlData.index[-1]))
    atlData.interpolate(inplace=True)
    atlData = atlData.resample('D').bfill()

    if r == '': pickle.dump(atlData.ix['2014-01-01':'2015-01-01'], open('atlDataAdvO.p','wb'))
    else: pickle.dump(atlData.ix['2014-01-01':'2015-01-01'], open('atlDataAdvO'+r+'.p','wb'))
    if r == '': pickle.dump(atlData.ix['2015-01-01':'2016-01-01'], open('atlDataAdvO2.p','wb'))
    else: pickle.dump(atlData.ix['2015-01-01':'2016-01-01'], open('atlDataAdvO'+r+'2.p','wb'))

"""
atlData = pickle.load(open('atlData.p','rb'))

# rolling mean
ma = pd.rolling_mean(atlData, 4)
mstd = pd.rolling_std(atlData, 4)

# Hodrick-Prescott filter
cycle, trend = sm.tsa.filters.hpfilter(atlData,270400)

fig, ax = plt.subplots(1,1,figsize=(11,5),sharey=False)
ax.plot(atlData.index, atlData, 'k', label='Historical')
ax.plot(ma.index, ma, 'b', label='Moving Avg')
ax.fill_between(mstd.index, ma-mstd, ma+mstd, color='y', alpha=0.2, label='Moving Std')
ax.plot(trend.index, trend, 'g', label='Trend')
ax.set_ylabel('Avg Occupancy')
ax.legend(loc='lower right')
ax.set_title('Weekly Average Occupancy in Atlanta, 2014')

fig.savefig('avgOccupancy.png')


from statsmodels.tsa.stattools import acf

# Auto-Correlation Function
lag_corr = acf(atlData.iloc[1:]) # lag by 1

fig, ax = plt.subplots(1,2,figsize=(11,5),sharey=True)
ax[0].plot(lag_corr, marker='o', linestyle='--', label='ACF')
ax[0].axhline(y=0)
ax[0].set_title('Original time-series')
ax[0].set_xlabel('Lags')
ax[0].set_ylabel('Correlation')
ax[0].legend(loc='upper right')

atlDataDiff = atlData - atlData.shift()
lag_corr = acf(atlDataDiff.iloc[1:]) # lag by 1
ax[1].plot(lag_corr, marker='o', linestyle='--', label='ACF')
ax[1].axhline(y=0)
ax[1].set_title('Transformed time-series')
ax[1].set_xlabel('Lags')
ax[1].legend(loc='upper right')

fig.savefig('ACF.png')


# ARIMA
model = sm.tsa.ARIMA(atlDataDiff.iloc[1:], order=(1,0,0)) # First-order autoregressive model
results = model.fit(disp=-1)
predict = results.predict()

fig, ax = plt.subplots(1,2,figsize=(11,5))
ax[0].plot(atlDataDiff.index, atlDataDiff,'y',label='Historical diff')
results.plot_predict(0,len(atlDataDiff)-1,dynamic=False,ax=ax[0],plot_insample=False)
ax[0].set_title('AR(1) forecast for $y_t$')
ax[0].set_ylabel('First-order difference in average occupancy')
labels = ax[0].get_xticklabels()
for label in labels:
    label.set_rotation(45)

ax[1].plot(atlData.index, atlData, 'y', label='Historical')
ax[1].plot(atlData.index, atlData + predict, 'r', label='forecast')
ax[1].set_title('AR(1) forecast projected onto $x_t=x_{t-1} + y_t$')
ax[1].set_ylabel('Average occupancy')
ax[1].legend(loc='upper right')
labels = ax[1].get_xticklabels()
for label in labels:
    label.set_rotation(45)

fig.savefig('ar_1.png')


# Different market segments, labeled by market experts
green = pickle.load(open('atlDataGREEN.p','rb')) # major hotels
black = pickle.load(open('atlDataBLACK.p','rb')) # within the perimeter of GREEN
red = pickle.load(open('atlDataRED.p','rb')) # outside the perimeter of GREEN
greenD = pickle.load(open('atlDataGREENDay.p','rb')) # major hotels
blackD = pickle.load(open('atlDataBLACKDay.p','rb')) # within the perimeter of GREEN
redD = pickle.load(open('atlDataREDDay.p','rb')) # outside the perimeter of GREEN

# Cross-correlation
bb, rr, gb, gr, bg, br, gg = [], [], [], [], [], [], []
bbD, rrD, gbD, grD, bgD, brD, ggD = [], [], [], [], [], [], []
lags = range(1,24)
for l in lags:
    bb.append(black.corr(black.shift(l)))
    rr.append(red.corr(red.shift(l)))
    gb.append(green.corr(black.shift(l)))
    gr.append(green.corr(red.shift(l)))
    bg.append(black.corr(green.shift(l)))
    br.append(black.corr(red.shift(l)))
    gg.append(green.corr(green.shift(l)))
    bbD.append(blackD.corr(blackD.shift(l)))
    rrD.append(redD.corr(redD.shift(l)))
    gbD.append(greenD.corr(blackD.shift(l)))
    grD.append(greenD.corr(redD.shift(l)))
    bgD.append(blackD.corr(greenD.shift(l)))
    brD.append(blackD.corr(redD.shift(l)))
   15 ggD.append(greenD.corr(greenD.shift(l)))
    
fig, ax = plt.subplots(2,3,figsize=(12,7))
ax[0][0].plot(green.index, green, 'g', label='GRN')
ax[0][0].plot(black.index, black, 'k', label='BLK')
ax[0][0].plot(red.index, red, 'r', label='RED')
ax[0][0].set_title('Weekly Avg Occupancy')
ax[0][0].legend(loc='upper center',ncol=3)
ax[0][0].set_ylabel('Average Occupancy')
ax[0][0].set_xticks([])

ax[0][1].plot(lags, bb, 'k--', linewidth=0.8, label='BLK => BLK')
ax[0][1].plot(lags, gb, 'g', label='GRN => BLK')
ax[0][1].set_title('Cross correlation (GRN => BLK)')
ax[0][1].legend(loc='upper center',ncol=3)
ax[0][1].set_xlabel('Lags in weeks')
ax[0][1].set_xticks([])

ax[0][2].plot(lags, rr, 'r--', linewidth=0.8, label='RED => RED')
ax[0][2].plot(lags, gr, 'g', label='GRN => RED')
ax[0][2].set_title('Cross correlation (GRN => RED)')
ax[0][2].legend(loc='upper center',ncol=3)
ax[0][2].set_xlabel('Lags in weeks')
ax[0][2].set_xticks([])

ax[1][0].plot(greenD.index, greenD, 'g', label='GRN')
ax[1][0].plot(blackD.index, blackD, 'k', label='BLK')
ax[1][0].plot(redD.index, redD, 'r', label='RED')
ax[1][0].set_title('Daily Avg Occupancy')
ax[1][0].legend(loc='upper center',ncol=3)
ax[1][0].set_ylabel('Average Occupancy')
labels = ax[1][0].get_xticklabels()
for label in labels:
    label.set_rotation(45)

ax[1][1].plot(lags, bbD, 'k--', linewidth=0.8, label='BLK => BLK')
ax[1][1].plot(lags, gbD, 'g', label='GRN => BLK')
ax[1][1].legend(loc='upper center',ncol=3)
ax[1][1].set_xlabel('Lags in days')

ax[1][2].plot(lags, rrD, 'r--', linewidth=0.8, label='RED => RED')
ax[1][2].plot(lags, grD, 'g', label='GRN => RED')
ax[1][2].legend(loc='upper center',ncol=3)
ax[1][2].set_xlabel('Lags in days')

#historical booking model
atlDataD = pickle.load(open('atlDataDay.p','rb')) # year 2014
atlDataD2 = pickle.load(open('atlDataDay2.p','rb')) # year 2015
dailyData = pd.concat([atlDataD, atlDataD2[1:]])

atlDataDiff = dailyData - dailyData.shift(14) # Forecast two weeks ahead
model = sm.tsa.ARIMA(atlDataDiff.iloc[14:], order=(3,0,0)) # First-order autoregressive model
results = model.fit(disp=-1) # trained on 2014 data
predict = results.predict(330, 414, dynamic=False) # index correcsponds to dailyData[330+14:400+14+1]
test_results = dailyData[330:428+1].reindex(predict.index)
test_results = test_results.add(predict,fill_value=0)

fig, ax = plt.subplots(1,2,figsize=(11,4), sharey=True)
ax[0].plot(test_results.index, dailyData[test_results.index], 'y', label='Historical')
ax[0].plot(test_results.index, test_results, 'r', linewidth=0.8, label='forecast')
ax[0].set_title('AR(3) for 14-day in advance prediction')
ax[0].set_ylabel('Average occupancy')
ax[0].legend(loc='lower center', ncol=2)
labels = ax[0].get_xticklabels()
for label in labels:
    label.set_rotation(45)


atlDataDiff = dailyData - dailyData.shift(30) # Forecast one month ahead
model = sm.tsa.ARIMA(atlDataDiff.iloc[30:], order=(3,0,0)) # First-order autoregressive model
results = model.fit(disp=-1) # trained on 2014 data
predict = results.predict(330, 430, dynamic=False) # index correcsponds to dailyData[330+14:400+14+1]
test_results = dailyData[330:460+1].reindex(predict.index)
test_results = test_results.add(predict,fill_value=0)
    
ax[1].plot(test_results.index, dailyData[test_results.index], 'y', label='Historical')
ax[1].plot(test_results.index, test_results, 'r', linewidth=0.8, label='forecast')
ax[1].set_title('AR(3) for 30-day in advance prediction')
ax[1].set_ylabel('Average occupancy')
ax[1].legend(loc='lower center', ncol=2)
labels = ax[1].get_xticklabels()
for label in labels:
    label.set_rotation(45)
fig, ax = plt.subplots(1,2,figsize=(11,4),sharey=True)
ax[0].plot(lags, ggD, 'g--', linewidth=0.8, label='GRN => GRN')
ax[0].plot(lags, bgD, 'k', label='BLK => GRN')
ax[0].set_title('Cross correlation (BLK => GRN)')
ax[0].legend(loc='lower left')
ax[0].set_xlabel('Lags in days')

ax[1].plot(lags, rrD, 'r--', linewidth=0.8, label='RED => RED')
ax[1].plot(lags, brD, 'k', label='BLK => RED')
ax[1].set_title('Cross correlation (BLK => RED)')
ax[1].legend(loc='lower left')
ax[1].set_xlabel('Lags in days')   


# Historical model
atlDataD = pickle.load(open('atlDataDay.p','rb')) # year 2014
atlDataD2 = pickle.load(open('atlDataDay2.p','rb')) # year 2015
atlDataDiff = atlDataD - atlDataD.shift()
dailyData = pd.concat([atlDataD, atlDataD2])
model = sm.tsa.ARIMA(atlDataDiff.iloc[1:], order=(7,0,0)) # First-order autoregressive model
results = model.fit(disp=-1) # trained on 2014 data
predict = results.predict()

fig, ax = plt.subplots(1,2,figsize=(11,4), sharey=True)
ax[0].plot(atlDataD.index, atlDataD, 'y', label='Historical')
#results.plot_predict('2014-12-01', '2016-01-01',ax=ax, plot_insample=False)
ax[0].plot(atlDataD.index, atlDataD + predict, 'r', linewidth=0.8, label='forecast')
ax[0].set_title('AR(7) forecast on training data')
ax[0].set_ylabel('Average occupancy')
ax[0].legend(loc='lower center', ncol=2)
labels = ax[0].get_xticklabels()
for label in labels:
    label.set_rotation(45)

predict2 = results.predict('2015-01-01','2016-01-01',dynamic=True)
test_results = pd.Series(atlDataD2.ix[0], index=atlDataD2.index)
test_results = test_results.add(predict2.cumsum(),fill_value=0)
ax[1].plot(atlDataD2.index, atlDataD2, 'y', label='Historical')
ax[1].plot(atlDataD2.index, test_results, 'b', linewidth=0.8, label='forecast')
ax[1].set_title('AR(7) forecast on testing data')
ax[1].legend(loc='lower center', ncol=2)
labels = ax[1].get_xticklabels()
for label in labels:
    label.set_rotation(45)
"""
