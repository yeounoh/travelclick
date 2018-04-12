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
    
    fr = open("daily_single_red_result.txt", 'w')
    fb = open("daily_single_blk_result.txt", 'w')
    for i in range(len(types)):
        fr.write(types[i]+"\n")
        fb.write(types[i]+"\n")
        for lookback in [4, 8]:
            for lookahead in [1, 2, 4, 8]:

                filter_cond1 = {'MARKET':mkt, 'MARKET_RANK':"RED", 'HOTEL_CLASS':types[i],'AVG_LEAD_TIME':0}
                filter_cond2 = {'MARKET':mkt, 'MARKET_RANK':"BLACK", 'HOTEL_CLASS':types[i],'AVG_LEAD_TIME':0}
                filter_cond3 = {'MARKET':mkt, 'MARKET_RANK':"RED", 'HOTEL_CLASS':types[i],'AVG_LEAD_TIME':1}
                filter_cond4 = {'MARKET':mkt, 'MARKET_RANK':"BLACK", 'HOTEL_CLASS':types[i],'AVG_LEAD_TIME':1}

                                
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
                result =forecast(booking[target], prebooking[target], booking_red[target], prebooking_red[target],lookback, lookahead, 'forecast_red_'+types[i]+'_one.png')
                if result is None:
                    continue
                fr.write(lookback+","+lookahead+","+str(result[0])+","+str(result[1])+","+str(result[2])+","+str(result[3])+"\n")

                hotel_idx = []
                booking_blk = loadHotels(filename,hotel_idx,attr_sum,attr_mean,filter_cond=filter_cond2,tframe=tf)
                booking_blk = booking_blk.ix['2014-01-01':'2016-01-01']
                booking_blk = booking_blk.reindex(pd.date_range(booking.index[0],booking.index[-1]))
                booking_blk.interpolate(inplace=True)
                prebooking_blk = loadHotels(filename,hotel_idx,attr_sum,attr_mean,filter_cond=filter_cond4,tframe=tf)
                prebooking_blk = prebooking_blk.ix['2014-01-01':'2016-01-01']
                prebooking_blk = prebooking_blk.reindex(pd.date_range(booking.index[0],booking.index[-1]))
                prebooking_blk.interpolate(inplace=True)
                result = forecast(booking[target], prebooking[target], booking_blk[target], prebooking_blk[target],lookback, lookahead, 'forecast_blk_'+types[i]+'_one.png')
                if result is None:
                    continue
                fb.write(lookback+","+lookahead+","+str(result[0])+","+str(result[1])+","+str(result[2])+","+str(result[3])+"\n")
                


    fr.close()                
    fb.close()
    
