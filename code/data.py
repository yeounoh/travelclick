import os
import numpy as np
import pandas as pd
import json
import pickle

"""
    Based on raw data format/schema.
    @OBSOLETE
    @param filename e.g., '/../data/booking_proc.csv'
    @param attr_sum, attr_mean defines attributes to be summed or averaged over resampling
    @param tframe e.g., 'W' for weekly and 'M' for monthly time-series resampling.
    @param filter_cond e.g., {'market':'San Francisco','k_property':11637}
    @return booking_data
"""
def loadRawData(filename,attr_sum,attr_mean,tframe='W',filter_cond={}):
    path = os.getcwd() + filename
    save_path = os.getcwd() + filename
    original_data = pd.read_csv(path)

    # Q: on a given day, how many rooms are occupied at what average price per hotel (in a city)?
    original_data = original_data.sort_values(by=['k_property','k_book_date'])
    original_data['k_property'] = pd.to_numeric(original_data['k_property'])

    # this needs to be integrated with date table
    date_table = pd.read_csv(os.getcwd() + '/../data/date.csv')
    date_table =date_table.set_index(['K_DATE'])
    original_data['book_date'] = original_data.join(date_table, on='k_book_date')['CALENDAR_DATE']
    original_data['book_date'] = pd.to_datetime(original_data['book_date'])

    booking_data = original_data #we probably need to deep-copy this
    for k,v in filter_cond.iteritems():
        print 'applying filter', k, '=', v
        booking_data = booking_data[booking_data[k] == v]
        #k = input('Enter hotel index (%d~%d): ' % (0,len(booking_data['k_property'].unique())-1))
        #k_prop = original_data['k_property'].unique()[k]
        #booking_data = original_data[original_data['k_property'] == k_prop]

    booking_data = original_data.set_index(['book_date'])
    bds_sum = booking_data.groupby(booking_data.index)[attr_sum].sum() #for a given day
    bds_mean = booking_data.groupby(booking_data.index)[attr_mean].mean()
    booking_data = pd.concat([bds_sum, bds_mean], axis=1)

    booking_data = booking_data.reindex(pd.date_range(booking_data.index[0],booking_data.index[-1]))
    booking_data.interpolate(inplace=True) # fll in the missing values

    bds_sum = booking_data[attr_sum].resample(tframe).sum() 
    bds_mean = booking_data[attr_mean].resample(tframe).mean()
    booking_data = pd.concat([bds_sum, bds_mean], axis=1)

    return booking_data

"""
    Based on Demand data format/schema, prepared with Marty's query.
    @param filename e.g., '/../data/booking_proc.csv'
    @param attr_sum, attr_mean defines attributes to be summed or averaged over resampling
    @param tframe e.g., 'W' for weekly time series
    @param filter_cond e.g., {'market':'San Francisco','k_property':11637}
    @return bookings array e.g., [bookings_11637, bookings_SF] 
"""
def loadData(filename,attr_sum,attr_mean,tframe='W',filter_cond={}):
    path = os.getcwd() + filename
    save_path = os.getcwd() + filename
    original_data = pd.read_csv(path,sep='|')

    # Q: on a given day, how many rooms are occupied at what average price per hotel (in a city)?
    original_data = original_data.sort_values(by=['PROPERTY_IDENTIFIER','STAY_DATE'])
    original_data['PROPERTY_IDENTIFIER'] = pd.to_numeric(original_data['PROPERTY_IDENTIFIER'])
    original_data['STAY_DATE'] = pd.to_datetime(original_data['STAY_DATE'])

    booking_data = original_data #we probably need to deep-copy this
    for k,v in filter_cond.iteritems():
        if k == 'AVG_LEAD_TIME':
            if v < 0:
                print 'applying filter', k, '<', -1 * v
                booking_data = booking_data[booking_data[k] < -1 * v]
            else:
                print 'applying filter', k, '>=', v
                booking_data = booking_data[booking_data[k] >= v]
        else:
            print 'applying filter', k, '=', v, 'in', booking_data[k].unique()
            if type(v) is list:
                booking_data = booking_data[booking_data[k].isin(v)]
            else:
                if '~' in v:
                    v = v.replace('~','')
                    booking_data = booking_data[~booking_data[k].isin([v])]
                else:
                    booking_data = booking_data[booking_data[k] == v]
                

    if booking_data.empty:
        return booking_data

    #booking_data = original_data.set_index(['STAY_DATE'])
    booking_data = booking_data.set_index(['STAY_DATE'])

    #filter based on date index
    #booking_data = booking_data.ix['2014-01-01':'2015-01-01'] # 2014 data only
    #print '2014-01-01 ~ 2015-01-01 has', len(booking_data.index), 'data points.'
    ##########################

    # additional statistics
    booking_data['COUNT'] = booking_data['PROPERTY_IDENTIFIER']
    bds_cnt = booking_data.groupby(booking_data.index)['COUNT'].count()
    # group by sum
    bds_sum = booking_data.groupby(booking_data.index)[attr_sum].sum() #for a given day
    # group by mean
    bds_mean = booking_data.groupby(booking_data.index)[attr_mean].mean()
    booking_data = pd.concat([bds_sum, bds_mean, bds_cnt], axis=1)

    # simple data cleaning
    booking_data = booking_data.reindex(pd.date_range(booking_data.index[0],booking_data.index[-1]))
    booking_data.interpolate(inplace=True) # flll in the missing values

    bds_cnt = booking_data['COUNT'].resample(tframe).sum()
    bds_sum = booking_data[attr_sum].resample(tframe).sum() 
    bds_mean = booking_data[attr_mean].resample(tframe).mean()
    booking_data = pd.concat([bds_sum, bds_mean, bds_cnt], axis=1)

    print 'data loaded & processed\n'

    return booking_data


'''
    Load occupancy time-series data for a randomly selected hotel
    @param filter_cond now extends to TYPE, which restricts the random selection space.
'''
def loadHotels(filename,hotel_idx,attr_sum,attr_mean,tframe='D',filter_cond={}):
    path = os.getcwd() + filename
    save_path = os.getcwd() + filename
    original_data = pd.read_csv(path,sep='|')

    # Q: on a given day, how many rooms are occupied at what average price per hotel (in a city)?
    original_data = original_data.sort_values(by=['PROPERTY_IDENTIFIER','STAY_DATE'])
    original_data['PROPERTY_IDENTIFIER'] = pd.to_numeric(original_data['PROPERTY_IDENTIFIER'])
    original_data['STAY_DATE'] = pd.to_datetime(original_data['STAY_DATE'])

    booking_data = original_data #we probably need to deep-copy this
    for k,v in filter_cond.iteritems():
        if k == 'AVG_LEAD_TIME':
            if v < 0:
                print 'applying filter', k, '<', -1 * v
                booking_data = booking_data[booking_data[k] < -1 * v]
            else:
                print 'applying filter', k, '>=', v
                booking_data = booking_data[booking_data[k] >= v]
        else:
            print 'applying filter', k, '=', v, 'in', booking_data[k].unique()
            if type(v) is list:
                booking_data = booking_data[booking_data[k].isin(v)]
            else:
                if '~' in v:
                    v = v.replace('~','')
                    booking_data = booking_data[~booking_data[k].isin([v])]
                else:
                    booking_data = booking_data[booking_data[k] == v]

    if booking_data.empty:
        return booking_data

    #select a hotel from the specified hotel category (TYPE)
    if hotel_idx == []:
        hotels = booking_data['PROPERTY_IDENTIFIER'].unique()
        if hotels != []:
            hotel_idx = np.random.choice(hotels,1)
    booking_data = booking_data[booking_data['PROPERTY_IDENTIFIER'].isin(hotel_idx)]

    #booking_data = original_data.set_index(['STAY_DATE'])
    booking_data = booking_data.set_index(['STAY_DATE'])

    # additional statistics
    booking_data['COUNT'] = booking_data['PROPERTY_IDENTIFIER']
    bds_cnt = booking_data.groupby(booking_data.index)['COUNT'].count()
    # group by sum
    bds_sum = booking_data.groupby(booking_data.index)[attr_sum].sum() #for a given day
    # group by mean
    bds_mean = booking_data.groupby(booking_data.index)[attr_mean].mean()
    booking_data = pd.concat([bds_sum, bds_mean, bds_cnt], axis=1)

    # simple data cleaning
    booking_data = booking_data.reindex(pd.date_range(booking_data.index[0],booking_data.index[-1]))
    booking_data.interpolate(inplace=True) # flll in the missing values

    bds_cnt = booking_data['COUNT'].resample(tframe).sum()
    bds_sum = booking_data[attr_sum].resample(tframe).sum() 
    bds_mean = booking_data[attr_mean].resample(tframe).mean()
    booking_data = pd.concat([bds_sum, bds_mean, bds_cnt], axis=1)

    print 'data loaded & processed\n'

    return booking_data
