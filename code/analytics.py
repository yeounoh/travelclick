import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt, mpld3
import statsmodels.api as sm
import seaborn as sb
sb.set_style('darkgrid')
import json
import pickle
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from pybrain.datasets import SequentialDataSet
from itertools import cycle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer
from sys import stdout


def logreg(tin,tout):
    return

"""
    Forecast the future (i.e., lookahead y days) using booking (look back x days) 
    and pre-booking data. Currently, the base learner is SVR -- need parameter sweep.
"""
def forecast(occGreen,prebookGreen,occTarget,prebookTarget,lookback,lookahead,title='forecast.png'):

    if occGreen.empty or prebookGreen.empty or occTarget.empty or prebookTarget.empty:
        return

    x = lookback
    y = lookahead

    # prepare input data 
    datain = np.zeros(shape=(len(occGreen[:-x]),x+y))
    datain_self = np.zeros(shape=(len(occTarget[:-x]),x+y))
    datain_prebook = np.zeros(shape=(len(occTarget[:-x]),y))
    datain_all = np.zeros(shape=(len(occGreen[:-x]),2*(x+y)))
    for i in range(x):
        datain[:,i] = occGreen[i:-x+i]
        datain_self[:,i] = occTarget[i:-x+i]
        datain_all[:,i] = occGreen[i:-x+i]
        datain_all[:,i+x+y] = occTarget[i:-x+i]
    for i in range(y):
        datain[:,x+i] = np.append(prebookGreen[x+i:-y+i],np.zeros(len(datain)-len(prebookGreen[x:-y])))
        datain_self[:,x+i] = np.append(prebookTarget[x+i:-y+i],np.zeros(len(datain_self)-len(prebookTarget[x:-y])))
        datain_prebook[:,i] = np.append(prebookTarget[x+i:-y+i],np.zeros(len(datain_prebook)-len(prebookTarget[x:-y])))
        datain_all[:,x+i] = np.append(prebookGreen[x+i:-y+i],np.zeros(len(datain)-len(prebookGreen[x:-y])))
        datain_all[:,x+i+x+y] = np.append(prebookTarget[x+i:-y+i],np.zeros(len(datain_self)-len(prebookTarget[x:-y])))
    dataout = occTarget[x+y-1:-1] # y days lookahead

    svr_rbf = SVR(kernel='rbf',C=1e1)
    if np.isnan(datain).any():
        return
    else:
        pred_grn = svr_rbf.fit(datain[:len(datain)/2+1], dataout[:len(datain)/2+1]).predict(datain)
    if np.isnan(datain_self).any():
        return
    else:
        pred_red = svr_rbf.fit(datain_self[:len(datain_self)/2+1], dataout[:len(datain_self)/2+1]).predict(datain_self)
    if np.isnan(datain_prebook).any():
        return
    else:
        pred_pre = svr_rbf.fit(datain_prebook[:len(datain_self)/2+1], dataout[:len(datain_prebook)/2+1]).predict(datain_prebook)
    if np.isnan(datain_all).any():
        return  
    else:
        pred_all = svr_rbf.fit(datain_all[:len(datain_all)/2+1], dataout[:len(datain_all)/2+1]).predict(datain_all)
    #print 'training.'
    #svr_poly = SVR(kernel='poly', C=1e1, degree=3)
    #pred_poly = svr_poly.fit(datain[:len(datain)/2+1], dataout[:len(datain)/2+1]).predict(datain)
    '''
    ds_grn = SequentialDataSet(x+y, 1)
    ds_self = SequentialDataSet(x+y , 1)
    ds_all = SequentialDataSet(2*(x+y), 1)
    for din, dout in zip(datain[:len(datain)/2], dataout[:len(datain)/2]):
        ds_grn.addSample(din, dout)
    for din, dout in zip(datain_self[:len(datain)/2], dataout[:len(datain)/2]):
        ds_self.addSample(din, dout)
    for din, dout in zip(datain_all[:len(datain)/2], dataout[:len(datain)/2]):
        ds_all.addSample(din, dout)
    
    net_grn = buildNetwork(x+y, x+y, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)
    net_red = buildNetwork(x+y, x+y, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)
    net_all = buildNetwork(2*(x+y), 2*(x+y), 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)
    
    
    trainer_grn = RPropMinusTrainer(net_grn, dataset=ds_grn)
    trainer_red = RPropMinusTrainer(net_red, dataset=ds_grn)
    trainer_all = RPropMinusTrainer(net_all, dataset=ds_all)
    #train_errors = []
    EPOCHS_PER_CYCLE = 5
    CYCLES = 100
    EPOCHS = EPOCHS_PER_CYCLE * CYCLES
    for i in xrange(CYCLES):
        trainer_grn.trainEpochs(EPOCHS_PER_CYCLE)
        trainer_red.trainEpochs(EPOCHS_PER_CYCLE)
        trainer_all.trainEpochs(EPOCHS_PER_CYCLE)
        #train_errors.append(trainer.testOnData())
        epoch = (i+1) * EPOCHS_PER_CYCLE
        #print "\r epoch {}/{}".format(epoch, EPOCHS)
        stdout.flush()
    #print "LTSM training error =", train_errors[-1]

    pred_netGrn = []
    pred_netRed = []
    pred_netAll = []
    for i in range(len(datain)):
        pred_netGrn.append(net_grn.activate(datain[i])[0])
        pred_netRed.append(net_red.activate(datain_self[i])[0])
        pred_netAll.append(net_all.activate(datain_all[i])[0])
    #print np.mean(abs(pred_net[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:])
    '''
    plt.figure()
    plt.hold('on')
    plt.scatter(occTarget.index, occTarget, s=2, c='m',label='Occupancy')
    #plt.plot(occTarget.index, occTarget, c='y',label='Occupancy')
    plt.plot(occTarget[x+y:].index, pred_pre[:-y], c='k', linewidth=1, label='SVRpre')
    plt.plot(occTarget[x+y:].index, pred_grn[:-y], c='g', linewidth=1, label='SVRgrn')
    plt.plot(occTarget[x+y:].index, pred_red[:-y], c='r', linewidth=1, label='SVRred')
    plt.plot(occTarget[x+y:].index, pred_all[:-y], c='b', linewidth=1, label='SVRall')
    #plt.plot(occTarget[x+y:].index, pred_netGrn[:-y], c='g', marker='*',linewidth=1, label='LTSM') 
    #plt.plot(occTarget[x+y:].index, pred_netRed[:-y], c='r', marker='*',linewidth=1, label='LTSM') 
    #plt.plot(occTarget[x+y:].index, pred_netAll[:-y], c='b', marker='*',linewidth=1, label='LTSM') 
    #plt.plot(occTarget[x+y:].index, pred_poly[:-y], c='r', label='SVRpoly')
    plt.ylabel('Occupancy')
    plt.title(title.replace('forecast_','').replace('.png','') + ', x:' + str(lookback) + ', y:' + str(lookahead))
    #plt.plot(tout[x:].index, pred_out2, 'ro')
    plt.legend([ 
        'Pre:%0.3f'%np.mean(abs(pred_pre[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:]), 
        'GRN:%0.3f'%np.mean(abs(pred_grn[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:]), 
        'RED:%0.3f'%np.mean(abs(pred_red[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:]), 
        'ALL:%0.3f'%np.mean(abs(pred_all[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:]),
        #'GRNnet:%0.3f'%np.mean(abs(pred_netGrn[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:]), 
        #'REDnet:%0.3f'%np.mean(abs(pred_netRed[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:]), 
        #'ALLnet:%0.3f'%np.mean(abs(pred_netAll[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:]),
        'Target'
        ]) 
    plt.hold('off')
    plt.savefig(title,inches='tight')

    plt.figure()
    plt.hold('on')
    plt.scatter(occTarget[len(datain)/2:].index, occTarget[len(datain)/2:], s=2, c='m',label='Occupancy')
    plt.plot(occTarget[x+y+len(datain)/2:].index, pred_pre[len(datain)/2:-y], c='k', linewidth=1, label='SVRpre')
    plt.plot(occTarget[x+y+len(datain)/2:].index, pred_red[len(datain)/2:-y], c='r', linewidth=1, label='SVRred')
    plt.plot(occTarget[x+y+len(datain)/2:].index, pred_all[len(datain)/2:-y], c='b', linewidth=1, label='SVRall')
    #plt.plot(occTarget[x+y+len(datain)/2:].index, pred_netAll[len(datain)/2:-y], c='b', marker='*',linewidth=1, label='LTSM') 
    plt.ylabel('Occupancy')
    plt.title(title.replace('forecast_','').replace('.png','') + ', x:' + str(lookback) + ', y:' + str(lookahead))
    #plt.plot(tout[x:].index, pred_out2, 'ro')
    plt.legend([ 
        'Pre:%f'%np.mean(abs(pred_pre[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:]), 
        'RED:%f'%np.mean(abs(pred_red[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:]), 
        'ALL:%0.3f'%np.mean(abs(pred_all[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:]),
        #'ALLnet:%0.3f'%np.mean(abs(pred_netAll[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:]),
        'Target'
        ]) 
    plt.hold('off')
    plt.savefig('Ronly_'+title,inches='tight')


    Eself_pre = np.mean(abs(pred_pre[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:]) 
    Egrn = np.mean(abs(pred_grn[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:]) 
    Eself_pre_hist = np.mean(abs(pred_red[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:]) 
    Eall = np.mean(abs(pred_all[len(datain)/2:-y]-occTarget[x+y+len(datain)/2:])/occTarget[x+y+len(datain)/2:])
    return (Eself_pre, Eself_pre_hist, Egrn, Eall)

def svr(tin,tout,title='svr.png'):
    #datain = zip(tin[:-3], tin[1:-2], tin[2:-1])
    #datain = zip(tin[:-8], tin[1:-7], tin[2:-6], tin[3:-5], tin[4:-4], tin[5:-3],tin[6:-2], tin[7:-1])
    #datain = zip(tin[:-12], tin[1:-11], tin[2:-10], tin[3:-9], tin[4:-8], tin[5:-7],tin[6:-6], tin[7:-5], tin[8:-4], tin[9:-3], tin[10:-2], tin[11:-1])
    datain = np.matrix([tin[:-16], tin[1:-15], tin[2:-14], tin[3:-13], tin[4:-12], tin[5:-11],tin[6:-10], tin[7:-9], tin[8:-8], tin[9:-7], tin[10:-6], tin[11:-5], tin[12:-4], tin[13:-3], tin[14:-2], tin[15:-1]]).T
    #dataout = tout[3:]
    #dataout = tout[8:]
    #dataout = tout[12:]
    dataout = tout[16:]
    svr_rbf = SVR(kernel='rbf', C=1e1, gamma=0.1)
    svr_poly = SVR(kernel='poly', C=1e1, degree=10)
    pred_out = svr_rbf.fit(datain[:len(datain)/2], dataout[:len(datain)/2]).predict(datain)
    pred_out2 = svr_poly.fit(datain[:len(datain)/2], dataout[:len(datain)/2]).predict(datain)

    total_error = 0.
    total_error2 = 0.
    for i in range(len(datain)):
        total_error += abs(svr_rbf.predict(datain[i])[0]-tout[i])/tout[i]
        total_error2 += abs(svr_poly.predict(datain[i])[0]-tout[i])/tout[i]

    print "total error (rbf) =", total_error / dataout.size
    print "total error (poly, d=10) = ", total_error2 / dataout.size
    

    fig = plt.figure()
    #tout[16:].plot(ax=ax, title='Occupancy')
    plt.hold("on")
    plt.plot(tout[16:].index, tout[16:], 'y', linewidth=1.5)
    plt.plot(tout[16:].index, pred_out, 'b+')
    plt.plot(tout[16:].index, pred_out2, 'ro')
    plt.legend(['Occupancy', 'SVR_rbf', 'SVR_poly'])
    fig.tight_layout()
    plt.savefig(title,inches='tight')

"""
    pass in differenced (i.e., stationary) series
    e.g.) data = bookings[arima_target].iloc[1:]
    order = (0,1,1) # basic exponential smoothing
    order = (0,1,0) # random walk
    order = (0,1,2) # damped Holt's model
"""
def arima(data, order=(0,1,1)):

    model = sm.tsa.ARIMA(data, order=order)
    results = model.fit(disp=-1)
    
    Y = results.predict(6)
    
    diff = np.array(Y) - np.array(data[6:]) 
    #print diff

    total_error = 0.
    for i in range(len(diff)):
        total_error += abs(diff[i])/data[i]
    
    print "total error =", total_error / diff.size

def crossCorr(X,Y,lag=0):
    return X.corr(Y.shift(lag))

def aic():
    return

'''
    input and output time-series are of the same length
    - assume weekly time series data, we use 3-month data to forecast
'''
def ltsmXY(tin, tout, title='ltsm.png'):

    #datain = zip(tin[:-3], tin[1:-2], tin[2:-1])
    #datain = zip(tin[:-8], tin[1:-7], tin[2:-6], tin[3:-5], tin[4:-4], tin[5:-3],tin[6:-2], tin[7:-1])
    #datain = zip(tin[:-12], tin[1:-11], tin[2:-10], tin[3:-9], tin[4:-8], tin[5:-7],tin[6:-6], tin[7:-5], tin[8:-4], tin[9:-3], tin[10:-2], tin[11:-1])
    datain = zip(tin[:-16], tin[1:-15], tin[2:-14], tin[3:-13], tin[4:-12], tin[5:-11],tin[6:-10], tin[7:-9], tin[8:-8], tin[9:-7], tin[10:-6], tin[11:-5], tin[12:-4], tin[13:-3], tin[14:-2], tin[15:-1])

    #dataout = tout[3:]
    #dataout = tout[8:]
    #dataout = tout[12:]
    dataout = tout[16:]

    #ds = SequentialDataSet(3, 1)
    #ds = SequentialDataSet(8, 1)
    #ds = SequentialDataSet(12, 1)
    ds = SequentialDataSet(16, 1)

    for x, y in zip(datain[:len(datain)/2], dataout[:len(datain)/2]):
        ds.addSample(x, y)


    # add layers until overfitting the training data
    #net = buildNetwork(3,5,1,hiddenclass=LSTMLayer, outputbias=False, recurrent=True)
    #net = buildNetwork(8, 8, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)
    #net = buildNetwork(12, 20, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)
    net = buildNetwork(16, 20, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

    
    trainer = RPropMinusTrainer(net, dataset=ds)
    train_errors = []
    EPOCHS_PER_CYCLE = 5
    CYCLES = 100
    EPOCHS = EPOCHS_PER_CYCLE * CYCLES
    for i in xrange(CYCLES):
        trainer.trainEpochs(EPOCHS_PER_CYCLE)
        train_errors.append(trainer.testOnData())
        epoch = (i+1) * EPOCHS_PER_CYCLE
        #print "\r epoch {}/{}".format(epoch, EPOCHS)
        stdout.flush()

    print "final error =", train_errors[-1]

    pred_out = []
    for i in range(len(datain)):
        pred_out.append(net.activate(datain[i]))
    
    fig = plt.figure()
    #tout[16:].plot(ax=ax, title='Occupancy')
    plt.plot(tout[16:].index, tout[16:], 'y', linewidth=1.5)
    plt.plot(tout[16:].index, pred_out, 'b+')
    plt.legend(['Occupancy', 'LTSM'])
    fig.tight_layout()
    plt.savefig(title,inches='tight')


def ltsm(data):
    from pybrain.datasets import SequentialDataSet
    from itertools import cycle
    
    datain = zip(data[:-6], data[1:-5], data[2:-4], data[3:-3], data[4:-2], data[5:-1])
    dataout = data[6:]
    ds = SequentialDataSet(6, 1)
    for x, y in zip(datain, dataout):
        ds.addSample(x, y)

    from pybrain.tools.shortcuts import buildNetwork
    from pybrain.structure.modules import LSTMLayer

    net = buildNetwork(6, 7, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

    from pybrain.supervised import RPropMinusTrainer
    from sys import stdout
    
    trainer = RPropMinusTrainer(net, dataset=ds)
    train_errors = []
    EPOCHS_PER_CYCLE = 5
    CYCLES = 100
    EPOCHS = EPOCHS_PER_CYCLE * CYCLES
    for i in xrange(CYCLES):
        trainer.trainEpochs(EPOCHS_PER_CYCLE)
        train_errors.append(trainer.testOnData())
        epoch = (i+1) * EPOCHS_PER_CYCLE
        #print "\r epoch {}/{}".format(epoch, EPOCHS)
        stdout.flush()

    print "final error =", train_errors[-1]

    '''
    plt.figure()
    plt.plot(range(0, EPOCHS, EPOCHS_PER_CYCLE), train_errors)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.show()
    '''

    test_error = 0.
    cnt = 0
    for sample, target in ds.getSequenceIterator(0):
        #print "sample = ",  sample
        #print "predicted next sample = %4.1f" % net.activate(sample)
        #print "actual next sample = %4.1f" % target
        test_error += abs(net.activate(sample) - target)
        cnt += 1
    test_error /= cnt 
    print "test (train) error =", test_error
    

# lamb=1600 for quarterly, 1500/4**4=6.25 for annual, 
# 1600*3**4=129600 for monthly data.
def hodrickPrescott(X):
    X_monthly = X.resample('M').mean()
    cycle, trend = sm.tsa.filters.hpfilter(X_monthly,129600)
    
    return (X_monthly, trend, cycle)

# does X causes Y?
def grangerCausality(X,Y,maxlag):
    results = sm.tsa.stattools.grangercausalitytests([Y,X],maxlag,verbose=True)
    for k, v in results.iteritems():
        print v[0]
    print results
    return results


"""
    Perform and rudimentary analytics on the loaded data; plot the results (also saved as jsons for web).
    I have put more advanced (lengthy) analytics implementations in a separate .py file.
    @param bookings from data.loadData()
    @param target attribute for time-series analysis
    @param save_path points to the folder to store figures
    @param postfix appended to the file name of each figure (before file extension)
    @return jsons list of json of each figure
"""
def oneSeries(bookings, target, save_path, postfix='', ):
    jsons = []
    save_path = os.getcwd() + save_path

    ############## rolling mean #############
    print 'Rolling mean filter on', target, 'column'
    ma = pd.rolling_mean(bookings[target], 20)
    mstd = pd.rolling_std(bookings[target], 20)
    fig, ax= plt.subplots(1,1,figsize=(8,8))
    ax.plot(bookings.index, bookings[target], 'k')
    ax.plot(ma.index, ma, 'b')
    ax.fill_between(mstd.index, ma-2*mstd, ma+2*mstd, color='b', alpha=0.2)
    fig.tight_layout()
    jsons.append( json.dumps(mpld3.fig_to_dict(fig)) )
    plt.savefig(save_path + target+'_ma_'+postfix+'.png',inches='tight')


    ############## Hodrick Prescott ###########
    print 'Hodrick Prescott filter on', target, 'column'
    (X_monthly, trend, cycle) = hodrickPrescott(bookings[target])
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    X_monthly = pd.DataFrame(X_monthly)
    X_monthly.columns = [target]
    X_monthly['cycle'] = cycle
    X_monthly['trend'] = trend
    X_monthly.plot(ax=ax,title='%s Hodrick Prescott analysis'%(target))
    fig.tight_layout()
    jsons.append( json.dumps(mpld3.fig_to_dict(fig)) )
    plt.savefig(save_path + target+'_hp_'+postfix+'.png',inches='tight')

    #first difference: transform non-stationary to stationary with zero mean
    bookings['First Difference'] = bookings[target] - bookings[target].shift()
    #log transform: the series grows exponentially -- flatens the data to a linear curve
    bookings['Natural Log'] = bookings[target].apply(lambda x: np.log(x))

    bookings['Original Variance'] = pd.rolling_var(bookings[target], 30, min_periods=None, freq=None, center=True)
    bookings['Log Variance'] = pd.rolling_var(bookings['Natural Log'], 30, min_periods=None, freq=None, center=True)
    bookings['Logged First Difference'] = bookings['Natural Log'] - bookings['Natural Log'].shift()

    fig, ax = plt.subplots(6, 1, figsize=(8,10))
    bookings[target].plot(ax=ax[0], title=target)
    bookings['First Difference'].plot(ax=ax[1],sharex=ax[0], title='First Difference')
    bookings['Natural Log'].plot(ax=ax[2], sharex=ax[0],title='Natural Log')
    bookings['Original Variance'].plot(ax=ax[3], sharex=ax[0],title='Original Variance')
    bookings['Log Variance'].plot(ax=ax[4], sharex=ax[0],title='Log Variance')
    bookings['Logged First Difference'].plot(ax=ax[5], sharex=ax[0],title='Logged First Difference')
    fig.tight_layout()
    jsons.append( json.dumps(mpld3.fig_to_dict(fig)) )
    plt.savefig(save_path + target+'_'+postfix+'.png',inches='tight')

    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(bookings[target].values, model='additive', freq=30)
    fig = plt.figure()
    fig = decomposition.plot()
    fig.tight_layout()
    jsons.append( json.dumps(mpld3.fig_to_dict(fig)) )
    plt.savefig(save_path + 'decompose_'+postfix+'.png',inches='tight')

    ######### stationary transformed timeseries ##########
    tf_target = 'Logged First Difference'
    print 'Now we look at the', tf_target, 'of', target
    ######################################################

    bookings['Lag 1'] = bookings[target].shift()
    bookings = bookings.replace([np.inf, -np.inf], np.nan)
    bookings.interpolate(inplace=True) # fill the missing values
    #bookings.dropna(how="all")
    #print bookings[[target, 'Lag 1']]
    #bookings['Lag 2'] = bookings[target].shift(2)
    #bookings['Lag 5'] = bookings[target].shift(5)
    #bookings['Lag 30'] = bookings[target].shift(30)

    from statsmodels.tsa.stattools import acf
    from statsmodels.tsa.stattools import pacf

    # systematically test for autocorrelation 1~40 days lag
    lag_correlations = acf(bookings[target].iloc[1:])
    lag_partial_correlations = pacf(bookings[target].iloc[1:])

    # see if the value at time and at any time prior up to 40 steps behind are significantly (>0.2) correlated.
    fig, ax = plt.subplots(2,1,figsize=(8,8))
    ax[0].plot(lag_correlations, marker='o', linestyle='--')
    ax[1].plot(lag_partial_correlations, marker='x',linestyle='--')
    fig.tight_layout()
    jsons.append( json.dumps(mpld3.fig_to_dict(fig)) )
    plt.savefig(save_path + 'acf_'+postfix+'.png',inches='tight')

    # tightly packed, and normal around 0 -- doesn't seem to be much correlation.
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    sb.jointplot(target, 'Lag 1', bookings, kind='reg', size=5, ax= ax)
    jsons.append( json.dumps(mpld3.fig_to_dict(fig)) )
    plt.savefig(save_path + target+'_lag1_'+postfix+'.png',inches='tight')

    arima_target = target #tf_target
    fig, ax = plt.subplots(3,1,figsize=(8,8))
    #ARIMA, used over the undifferenced series. If we want to predict the first difference (day-to-day move)?
    model = sm.tsa.ARIMA(bookings[arima_target].iloc[1:], order=(0,1,1))
    results = model.fit(disp=-1)
    ax[0] = bookings[arima_target].iloc[0:].plot(ax=ax[0])
    results.plot_predict(48,'2017-02-01',dynamic=True,ax=ax[0],plot_insample=False)
    #bookings['Forecast'] = results.fittedvalues
    #bookings[[target, 'Forecast']].plot(ax=ax[0],title='ARIMA on %s' % target)

    #ARIMA, used over the undifferenced series. If we want to predict the first difference (day-to-day move)?
    model = sm.tsa.ARIMA(bookings[target].iloc[1:], order=(0,1,1))
    results = model.fit(disp=-1)
    ax[1] = bookings[target].iloc[0:].plot(ax=ax[1])
    results.plot_predict(48,'2017-02-01',dynamic=True,ax=ax[1],plot_insample=False)
    #bookings['Forecast'] = results.fittedvalues
    #bookings[[target, 'Forecast']].plot(ax=ax[0],title='ARIMA on %s' % target)

    #Exponential smoothing model
    model = sm.tsa.ARIMA(bookings[target].iloc[1:], order=(0,1,1))
    results = model.fit(disp=-1)
    #ax[1] = bookings[target].iloc[0:].plot(ax=ax[1])
    #results.plot_predict('2015-07-01','2015-12-31',dynamic=True,ax=ax[1],plot_insample=False)
    bookings['Fitted'] = results.fittedvalues
    bookings[[target, tf_target, 'Fitted']].plot(ax=ax[2],sharex=ax[0])

    fig.tight_layout()
    jsons.append( json.dumps(mpld3.fig_to_dict(fig)) )
    plt.savefig(save_path + 'arima_'+postfix+'.png',inches='tight')

    return jsons


def twoSeries(bookings1, bookings2, target, save_path, postfix='', ):
    jsons = []
    save_path = os.getcwd() + save_path

    ############## rolling mean #############
    print 'Rolling mean filter on', target, 'column'
    ma1 = pd.rolling_mean(bookings1[target], 10)
    mstd1 = pd.rolling_std(bookings1[target], 10)
    ma2 = pd.rolling_mean(bookings2[target], 10)
    mstd2 = pd.rolling_std(bookings2[target], 10)
    fig, ax= plt.subplots(1,1,figsize=(8,8))
    ax.plot(bookings1.index, bookings1[target], 'k')
    ax.plot(bookings2.index, bookings2[target], 'g')
    ax.plot(ma1.index, ma1, 'b')
    ax.plot(ma2.index, ma2, 'r')
    ax.fill_between(mstd1.index, ma1-2*mstd1, ma1+2*mstd1, color='b', alpha=0.2)
    ax.fill_between(mstd2.index, ma2-2*mstd2, ma2+2*mstd2, color='b', alpha=0.2)
    fig.tight_layout()
    jsons.append( json.dumps(mpld3.fig_to_dict(fig)) )
    plt.savefig(save_path + target+'_ma_'+postfix+'.png',inches='tight')

    #first difference: transform non-stationary to stationary with zero mean
    bookings1['First Difference'] = bookings1[target] - bookings1[target].shift()
    #log transform: the series grows exponentially -- flatens the data to a linear curve
    bookings1['Natural Log'] = bookings1[target].apply(lambda x: np.log(x))

    bookings1['Original Variance'] = pd.rolling_var(bookings1[target], 30, min_periods=None, freq=None, center=True)
    bookings1['Log Variance'] = pd.rolling_var(bookings1['Natural Log'], 30, min_periods=None, freq=None, center=True)
    bookings1['Logged First Difference'] = bookings1['Natural Log'] - bookings1['Natural Log'].shift()

    #first difference: transform non-stationary to stationary with zero mean
    bookings2['First Difference'] = bookings2[target] - bookings2[target].shift()
    #log transform: the series grows exponentially -- flatens the data to a linear curve
    bookings2['Natural Log'] = bookings2[target].apply(lambda x: np.log(x))

    bookings2['Original Variance'] = pd.rolling_var(bookings2[target], 30, min_periods=None, freq=None, center=True)
    bookings2['Log Variance'] = pd.rolling_var(bookings2['Natural Log'], 30, min_periods=None, freq=None, center=True)
    bookings2['Logged First Difference'] = bookings2['Natural Log'] - bookings2['Natural Log'].shift()

    ######### stationary transformed timeseries ##########
    tf_target = 'Logged First Difference'
    print 'Now we look at the', tf_target, 'of', target
    ######################################################

    bookings1 = bookings1.replace([np.inf, -np.inf], np.nan)
    bookings1.interpolate(inplace=True) # fill the missing values
    bookings2 = bookings2.replace([np.inf, -np.inf], np.nan)
    bookings2.interpolate(inplace=True) # fill the missing values

    from statsmodels.tsa.stattools import acf
    from statsmodels.tsa.stattools import pacf

    # systematically test for autocorrelation 1~40 days lag
    lag_correlations1 = acf(bookings1[target].iloc[1:])
    lag_partial_correlations1 = pacf(bookings1[target].iloc[1:])

    # see if the value at time and at any time prior up to 40 steps behind are significantly (>0.2) correlated.
    fig, ax = plt.subplots(2,1,figsize=(8,8))
    ax[0].plot(lag_correlations1, marker='o', linestyle='--')
    ax[1].plot(lag_partial_correlations1, marker='x',linestyle='--')
    fig.tight_layout()
    jsons.append( json.dumps(mpld3.fig_to_dict(fig)) )
    plt.savefig(save_path + 'acf_'+postfix+'1.png',inches='tight')

    lag_correlations2 = acf(bookings2[target].iloc[1:])
    lag_partial_correlations2 = pacf(bookings2[target].iloc[1:])

    # see if the value at time and at any time prior up to 40 steps behind are significantly (>0.2) correlated.
    fig, ax = plt.subplots(2,1,figsize=(8,8))
    ax[0].plot(lag_correlations2, marker='o', linestyle='--')
    ax[1].plot(lag_partial_correlations2, marker='x',linestyle='--')
    fig.tight_layout()
    jsons.append( json.dumps(mpld3.fig_to_dict(fig)) )
    plt.savefig(save_path + 'acf_'+postfix+'2.png',inches='tight')

    return jsons


def advancedBookingModel(bookings, lag):
    # booking curve
    return


