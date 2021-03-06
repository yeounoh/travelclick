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
from writeHTML import writeHTML      
from data import loadRawData, loadHotels,loadData
from analytics import grangerCausality, hodrickPrescott, crossCorr, oneSeries, ltsm, arima, ltsmXY, twoSeries, svr, forecast, logreg


if __name__ == "__main__":
    filename = '/../data/TC_Market_DMDData.dsv'
    #filename = '/../data/sample.csv' # for testing

    markets = ['Atlanta', 'Los Angeles', 'San Francisco', 'New York City', 'Rotterdam']
    #types = ['Upper Upscale', 'Upper Midscale', 'Upscale', 'Economy', 'Luxury', 'Midscale']
    types = ['Economy', 'Luxury', 'Midscale']
    ranks = ['GREEN', 'RED', 'BLACK']
    
    attr_sum = []
    attr_mean = ['AVG_ROOM_RATE','OCCUPANCY']
    attr = ['AVG_ROOM_RATE','OCCUPANCY'] #'STAY_DATE'

    tframes = ['W']
    tf = tframes[0]
    # target attribute should be part of attr (also part of attr_sum or attr_mean)
    target = 'OCCUPANCY'
    #target = 'COUNT'
    
    ####################################################
    ####            Forecast with Green             ####
    ####################################################
    mkt = 'Atlanta' #'San Francisco' #'New York City'
    
    filter_cond = {'MARKET':mkt, 'MARKET_RANK':'GREEN','AVG_LEAD_TIME':0}
    booking = loadData(filename,attr_sum,attr_mean,filter_cond=filter_cond,tframe=tf)
    booking = booking.ix['2014-01-01':'2016-01-01']
    booking = booking.reindex(pd.date_range(booking.index[0],booking.index[-1]))
    booking.interpolate(inplace=True)
    
    filter_cond = {'MARKET':mkt, 'MARKET_RANK':'GREEN','AVG_LEAD_TIME':1}
    prebooking = loadData(filename,attr_sum,attr_mean,filter_cond=filter_cond,tframe=tf)
    prebooking = prebooking.ix['2014-01-01':'2016-01-01']
    prebooking = prebooking.reindex(pd.date_range(booking.index[0],booking.index[-1]))
    prebooking.interpolate(inplace=True)
    
    fr = open("daily_htype_red_result.txt", 'w')
    fb = open("daily_htype_blk_result.txt", 'w')
    for i in range(len(types)):
        fr.write(types[i]+"\n")
        fb.write(types[i]+"\n")
        for lookback in [4, 8]:
            for lookahead in [1,2,4,8]:

                filter_cond1 = {'MARKET':mkt, 'MARKET_RANK':"RED", 'HOTEL_CLASS':types[i],'AVG_LEAD_TIME':0}
                filter_cond2 = {'MARKET':mkt, 'MARKET_RANK':"BLACK", 'HOTEL_CLASS':types[i],'AVG_LEAD_TIME':0}
                filter_cond3 = {'MARKET':mkt, 'MARKET_RANK':"RED", 'HOTEL_CLASS':types[i],'AVG_LEAD_TIME':1}
                filter_cond4 = {'MARKET':mkt, 'MARKET_RANK':"BLACK", 'HOTEL_CLASS':types[i],'AVG_LEAD_TIME':1}

                '''                
                # select one hotel at random from each hotel type
                hotel_idx = []
                booking_red = loadHotels(filename,hotel_idx,attr_sum,attr_mean,filter_cond=filter_cond1,tframe=tf)
                
                booking_red = booking_red.ix['2014-01-01':'2016-01-01']
                booking_red = booking_red.reindex(pd.date_range(booking.index[0],booking.index[-1]))
                booking_red.interpolate(inplace=True)
                prebooking_red = loadHotels(filename,hotel_idx,attr_sum,attr_mean,filter_cond=filter_cond3,tframe=tf)
                prebooking_red = prebooking_red.ix['2014-01-01':'2016-01-01']
                prebooking_red = prebooking_red.reindex(pd.date_range(booking.index[0],booking.index[-1]))
                prebooking_red.interpolate(inplace=True)
                forecast(booking[target], prebooking[target], booking_red[target], prebooking_red[target],lookback, lookahead, 'forecast_red_'+types[i]+'_one.png')

                hotel_idx = []
                booking_blk = loadHotels(filename,hotel_idx,attr_sum,attr_mean,filter_cond=filter_cond2,tframe=tf)
                booking_blk = booking_blk.ix['2014-01-01':'2016-01-01']
                booking_blk = booking_blk.reindex(pd.date_range(booking.index[0],booking.index[-1]))
                booking_blk.interpolate(inplace=True)
                prebooking_blk = loadHotels(filename,hotel_idx,attr_sum,attr_mean,filter_cond=filter_cond4,tframe=tf)
                prebooking_blk = prebooking_blk.ix['2014-01-01':'2016-01-01']
                prebooking_blk = prebooking_blk.reindex(pd.date_range(booking.index[0],booking.index[-1]))
                prebooking_blk.interpolate(inplace=True)
                forecast(booking[target], prebooking[target], booking_blk[target], prebooking_blk[target],lookback, lookahead, 'forecast_blk_'+types[i]+'_one.png')
                '''

                # per hotel type 
                booking_red = loadData(filename,attr_sum,attr_mean,filter_cond=filter_cond1,tframe=tf)
                booking_red = booking_red.ix['2014-01-01':'2016-01-01']
                booking_red = booking_red.reindex(pd.date_range(booking.index[0],booking.index[-1]))
                booking_red.interpolate(inplace=True)
                prebooking_red = loadData(filename,attr_sum,attr_mean,filter_cond=filter_cond3,tframe=tf)
                prebooking_red = prebooking_red.ix['2014-01-01':'2016-01-01']
                prebooking_red = prebooking_red.reindex(pd.date_range(booking.index[0],booking.index[-1]))
                prebooking_red.interpolate(inplace=True)
                result = forecast(booking[target], prebooking[target], booking_red[target], prebooking_red[target], lookback, lookahead, 'forecast_RED_'+types[i]+'_'+str(lookback)+'_'+str(lookahead)+'.png')
                if result is None:
                    continue
                fr.write(lookback+","+lookahead+","+str(result[0])+","+str(result[1])+","+str(result[2])+","+str(result[3])+"\n")

                booking_blk = loadData(filename,attr_sum,attr_mean,filter_cond=filter_cond2,tframe=tf)
                booking_blk = booking_blk.ix['2014-01-01':'2016-01-01']
                booking_blk = booking_blk.reindex(pd.date_range(booking.index[0],booking.index[-1]))
                booking_blk.interpolate(inplace=True)
                prebooking_blk = loadData(filename,attr_sum,attr_mean,filter_cond=filter_cond4,tframe=tf)
                prebooking_blk = prebooking_blk.ix['2014-01-01':'2016-01-01']
                prebooking_blk = prebooking_blk.reindex(pd.date_range(booking.index[0],booking.index[-1]))
                prebooking_blk.interpolate(inplace=True)
                result = forecast(booking[target], prebooking[target], booking_blk[target], prebooking_blk[target],lookback, lookahead, 'forecast_BLK_'+types[i]+'_'+str(lookback)+'_'+str(lookahead)+'.png')
                if result is None:
                    continue
                fb.write(lookback+","+lookahead+","+str(result[0])+","+str(result[1])+","+str(result[2])+","+str(result[3])+"\n")

    fr.close()                
    fb.close()
    

    fr = open("daily_red_result.txt", 'w')
    fb = open("daily_blk_result.txt", 'w')

    filter_cond1 = {'MARKET':mkt, 'MARKET_RANK':"RED", 'AVG_LEAD_TIME':0}
    filter_cond2 = {'MARKET':mkt, 'MARKET_RANK':"BLACK", 'AVG_LEAD_TIME':0}

    booking_red = loadData(filename,attr_sum,attr_mean,filter_cond=filter_cond1,tframe=tf)
    booking_red = booking_red.ix['2014-01-01':'2016-01-01']
    booking_red = booking_red.reindex(pd.date_range(booking.index[0],booking.index[-1]))
    booking_red.interpolate(inplace=True)
    prebooking_red = loadData(filename,attr_sum,attr_mean,filter_cond=filter_cond3,tframe=tf)
    prebooking_red = prebooking_red.ix['2014-01-01':'2016-01-01']
    prebooking_red = prebooking_red.reindex(pd.date_range(booking.index[0],booking.index[-1]))
    prebooking_red.interpolate(inplace=True)
    result =forecast(booking[target], prebooking[target], booking_red[target], prebooking_red[target], lookback, lookahead, 'forecast_RED_'+str(lookback)+'_'+str(lookahead)+'.png')
    fr.write(""+str(result[0])+","+str(result[1])+","+str(result[2])+","+str(result[3])+"\n")

    booking_blk = loadData(filename,attr_sum,attr_mean,filter_cond=filter_cond2,tframe=tf)
    booking_blk = booking_blk.ix['2014-01-01':'2016-01-01']
    booking_blk = booking_blk.reindex(pd.date_range(booking.index[0],booking.index[-1]))
    booking_blk.interpolate(inplace=True)
    prebooking_blk = loadData(filename,attr_sum,attr_mean,filter_cond=filter_cond4,tframe=tf)
    prebooking_blk = prebooking_blk.ix['2014-01-01':'2016-01-01']
    prebooking_blk = prebooking_blk.reindex(pd.date_range(booking.index[0],booking.index[-1]))
    prebooking_blk.interpolate(inplace=True)
    result =forecast(booking[target], prebooking[target], booking_blk[target], prebooking_blk[target],lookback, lookahead, 'forecast_BLK_'+str(lookback)+'_'+str(lookahead)+'.png')
    fb.write(""+str(result[0])+","+str(result[1])+","+str(result[2])+","+str(result[3])+"\n")
         
    fr.close()                
    fb.close()
    ####################################################

    """
    filter_cond = {'MARKET':mkt,'MARKET_RANK':"GREEN",'AVG_LEAD_TIME':0}
    bookings1 = loadData(filename,attr_sum,attr_mean,filter_cond=filter_cond,tframe=tf)
    #bookings1_train = bookings1.ix['2014-01-01':'2015-01-01']
    #bookings1_test = bookings1.ix['2015-01-02':'2016-01-01']

    #jsons = oneSeries(bookings1, target, '/../plot/',postfix='SF-green')
    #ltsm(bookings1[target])
    #arima(bookings1[target])
    

    filter_cond = {'MARKET':mkt,'MARKET_RANK':"RED",'AVG_LEAD_TIME':0}
    bookings2 = loadData(filename,attr_sum,attr_mean,filter_cond=filter_cond,tframe=tf)
    #bookings2_train = bookings2.ix['2014-01-01':'2015-01-01']
    #bookings2_test = bookings2.ix['2015-01-02':'2016-01-01']
    #jsons = oneSeries(bookings2, target, '/../plot/',postfix='SF-red')
    #arima(bookings2[target])
    print 'Atlanta, red -> red'
    svr(bookings2[target],bookings2[target],'svr_rr.png')
    ltsmXY(bookings2[target],bookings2[target],'ltsm_rr.png')
    print 'Atlanta, green -> red'
    svr(bookings1[target],bookings2[target],'svr_gr.png')
    ltsmXY(bookings1[target],bookings2[target],'ltsm_gr.png')

    filter_cond = {'MARKET':mkt, 'MARKET_RANK':"~GREEN",'AVG_LEAD_TIME':0}
    bookings3 = loadData(filename,attr_sum,attr_mean,filter_cond=filter_cond,tframe=tf)
    #twoSeries(bookings1, bookings2, target, '/../plot/',postfix='SF-green-red')
    """
    """            
    for tf in tframes:
        print 'tframe=', tf
        for m in markets:
            if m is 'Atlanta':
                continue
            print 'Analyzing market ', m

            red_red = []
            grn_red = []
            rb_rb = []
            grn_rb = []
            max_l = 12
            for l in range(1,max_l):
                bookings_type = []

                # we compare before and after l weeks
                delta = l * 7

                # Green market segment
                filter_t = {'MARKET':m, 'MARKET_RANK':"GREEN", 'AVG_LEAD_TIME':delta}
                bookings_type.append(loadData(filename,attr_sum,attr_mean,filter_cond=filter_t,tframe=tf))
                filter_t = {'MARKET':m, 'MARKET_RANK':"GREEN", 'AVG_LEAD_TIME':-1 * delta}
                bookings_type.append(loadData(filename,attr_sum,attr_mean,filter_cond=filter_t,tframe=tf))
                '''
                # Red market segment
                filter_t = {'MARKET':m, 'MARKET_RANK':"RED", 'AVG_LEAD_TIME':delta}
                bookings_type.append(loadData(filename,attr_sum,attr_mean,filter_cond=filter_t,tframe=tf))
                filter_t = {'MARKET':m, 'MARKET_RANK':"RED", 'AVG_LEAD_TIME':-1 * delta}
                bookings_type.append(loadData(filename,attr_sum,attr_mean,filter_cond=filter_t,tframe=tf))

                # Red and Black market segment
                filter_t = {'MARKET':m, 'MARKET_RANK':["RED","BLACK"], 'AVG_LEAD_TIME':delta}
                bookings_type.append(loadData(filename,attr_sum,attr_mean,filter_cond=filter_t,tframe=tf))
                filter_t = {'MARKET':m, 'MARKET_RANK':["RED","BLACK"], 'AVG_LEAD_TIME':-1 * delta}
                bookings_type.append(loadData(filename,attr_sum,attr_mean,filter_cond=filter_t,tframe=tf))

                corr = crossCorr(bookings_type[2][target],bookings_type[3][target],0)
                red_red.append(corr)
                
                corr = crossCorr(bookings_type[0][target],bookings_type[3][target],0)
                grn_red.append(corr)
                '''
                # Non-Green market segment
                filter_t = {'MARKET':m, 'MARKET_RANK':"~GREEN",'AVG_LEAD_TIME':delta}
                bdata = loadData(filename,attr_sum,attr_mean,filter_cond=filter_t,tframe=tf)
                bookings_type.append(bdata)
                filter_t = {'MARKET':m, 'MARKET_RANK':"~GREEN",'AVG_LEAD_TIME':-1 * delta}
                bdata = loadData(filename,attr_sum,attr_mean,filter_cond=filter_t,tframe=tf)
                bookings_type.append(bdata)
                
                corr = crossCorr(bookings_type[2][target],bookings_type[3][target],0)
                rb_rb.append(corr)

                corr = crossCorr(bookings_type[0][target],bookings_type[3][target],0)
                grn_rb.append(corr)
                
            #print 'l, r-r, g-r, rb-rb, g-rb'
            print 'l, rb-rb, g-rb'
            for l in range(1,max_l):
                #print l, ',', red_red[l-1], ',', grn_red[l-1], ',', rb_rb[l-1], ',', grn_rb[l-1] 
                print l, ',', rb_rb[l-1], ',', rb_rb[l-1]
    """
    """ 
    for tf in tframes:
        print 'tframe=', tf
        bookings_type = []
        for m in markets:
            if m is not 'Atlanta':
                continue
            # Q: does green lead red?
            #filter_t = {'MARKET':m, 'MARKET_RANK':"GREEN", 'AVG_LEAD_TIME':5}
            filter_t = {'MARKET':m, 'MARKET_RANK':"GREEN"}
            bookings_type.append(loadData(filename,attr_sum,attr_mean,filter_cond=filter_t,tframe=tf))
            #filter_t = {'MARKET':m, 'MARKET_RANK':"RED", 'AVG_LEAD_TIME':5}
            filter_t = {'MARKET':m, 'MARKET_RANK':"RED"}
            bookings_type.append(loadData(filename,attr_sum,attr_mean,filter_cond=filter_t,tframe=tf))
            #filter_t = {'MARKET':m, 'MARKET_RANK':["RED","BLACK"], 'AVG_LEAD_TIME':5}
            filter_t = {'MARKET':m, 'MARKET_RANK':["RED","BLACK"]}
            bookings_type.append(loadData(filename,attr_sum,attr_mean,filter_cond=filter_t,tframe=tf))
            '''
            green_ln = bookings_type[0][target].apply(lambda x: np.log(x))
            green_lnd = green_ln - green_ln.shift()
            red_ln = bookings_type[1][target].apply(lambda x: np.log(x))
            red_lnd = red_ln - red_ln.shift()
            rb_ln = bookings_type[2][target].apply(lambda x: np.log(x))
            rb_lnd = rb_ln - rb_ln.shift()
            '''
            print '\nloading market:', m, 'for asking "Does Green lead Red?'
            for l in range(-48,48):
                #corr = crossCorr(green_lnd,red_lnd,l)
                corr = crossCorr(bookings_type[0][target],bookings_type[1][target],l)
                print 'lag:', l, 'corr:', corr

            print '\nloading market:', m, 'for asking "Does Green lead Red+Black?'
            for l in range(-48,48):
                #corr = crossCorr(green_lnd,rb_lnd,l)
                corr = crossCorr(bookings_type[0][target],bookings_type[2][target],l)
                print 'lag:', l, 'corr:', corr
            #print 'granger causality test (green -> red)?'
            #grangerCausality(bookings_type[0][target],bookings_type[1][target],30)

        '''
        print 'loading market:', m
        for t in types:
            print 'loading hotel type:', t
            #filter_t = {'MARKET':m, 'HOTEL_CLASS':t}
            filter_t = {'MARKET':m, 'HOTEL_CLASS':t, 'AVG_LEAD_TIME':5}
            bookings_type.append(loadData(filename,attr_sum,attr_mean,filter_cond=filter_t))
        #pickle.dump( bookings_type, open(os.getcwd()+'/../data/bookings_type_'+m+'.p','wb') )
        #bookings_type = pickle.load(open(os.getcwd()+'/../data/bookings_type_'+m+'.p','rb'))

        for i in range(len(types)):
            for j in range(i+1,len(types)):
                print '###', types[i], types[j], 'cross-correlation', '###'
                for l in range(-10,10):
                    corr = crossCorr(bookings_type[i][target],bookings_type[j][target],l) 
                    print 'lag:', l, 'corr:', corr
        '''
    """            
             
        # maxlag
        #grangerCausality(bookings_lux[target],bookings_econ[target],-1)
        #grangerCausality(bookings_econ[target],bookings_lux[target],-1)

    # Generate index.html file with the results
    # the html layout and the visualizations need more work (not the core, and tedious) 
    # -- but we have the jsons to play with.
    ##writeHTML(jsons)
