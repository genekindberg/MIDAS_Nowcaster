#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Gene Kindberg-Hanlon
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tabulate import tabulate
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

pd.options.mode.chained_assignment = None 

class OptimAR:
    def __init__(self, GDP):
        self.GDP = GDP

        
    def Forecastperf(self, ARt): # not out of sample AR forecast evaluation
        self.ARt = ARt
        GDP = self.GDP
        length = len(GDP)-ARt
        X = np.zeros(shape=(length,ARt))
        # create various lag lengths
        for ii in range(1,ARt+1):
            X[0:,ii-1] = GDP[ARt-ii:-ii].T 
        
        Y = GDP[ARt:].reshape(-1,1)
        # create fitted values and test RMSE
        Fit_val = np.zeros(shape=(length,ARt))
        RMSE = np.zeros(shape=(1,ARt))
        for ii in range(1,ARt+1):
            Fit_val[0:,ii-1] = LinearRegression().fit(X[0:,0:ii].reshape(-1,ii), Y).predict(X[0:,0:ii].reshape(-1,ii)).T
            RMSE[0,ii-1] = np.sqrt(np.average(np.square(Y-Fit_val[0:,ii-1])))
        
        self.RMSE = RMSE
        self.Fit_val = Fit_val
        self.BestAR = RMSE.argmin() # location of lowest
            
    def PlotBest(self):
        #something
        Y = self.GDP[self.ARt:]
        Y_fit = self.Fit_val[0:,self.BestAR]
        fig2 = plt.figure(figsize=(15,15))
        ax1 = fig2.add_subplot(111)
        ax1.plot(range(0,len(Y)), Y,marker='', color='olive', linewidth=2)
        ax1.plot(range(0,len(Y)), Y_fit, marker='', color='blue', linewidth=2)
        ax1.legend(['Fitted', 'data'], loc='upper left')
        plt.show()        

class OptimARoos: ## Out of sample forecast evaluation
    def __init__(self, GDP):
        self.GDP = GDP

        
    def Forecastperf(self, ARt, skip):
        self.ARt = ARt
        self.skip = skip
        GDP = self.GDP
        length = len(GDP)-ARt
        X = np.zeros(shape=(length,ARt))
        # create various lag lengths
        for ii in range(1,ARt+1):
            X[0:,ii-1] = GDP[ARt-ii:-ii].T 
        
        Y = GDP[ARt:].reshape(-1,1)
        # create fitted values and test RMSE
        
        Fit_val = np.zeros(shape=(length-skip+1,ARt)) # plus 1 for forecast peiod
        RMSE = np.zeros(shape=(1,ARt))
        
        for ii in range(1,ARt+1):
            for jj in range(skip, length):
                Temp = LinearRegression().fit(X[0:jj,0:ii].reshape(-1,ii), Y[0:jj,0])
                Fit_val[jj-skip:jj+1-skip,ii-1] = Temp.predict(X[jj:jj+1,0:ii].reshape(-1,ii)).T
            RMSE[0,ii-1] = np.sqrt(np.average(np.square(Y[skip:,0]-Fit_val[0:-1,ii-1])))
            Fit_val[length-skip:length-skip+1,ii-1] = Temp.predict(Y[-ii:,:].reshape(-1,ii)).T # forecast quarter for each lag
            
        self.RMSE = RMSE
        self.Fit_val = Fit_val
        self.BestAR = RMSE.argmin() # location of lowest
        self.OptimFit = Fit_val[0:,self.BestAR]
        self.OptimRMSE = RMSE[0,self.BestAR]
        
    def PlotBest(self):
        #something
        Y = self.GDP[self.ARt:]
        Y = Y[self.skip:]
        Y_fit = self.Fit_val[0:,self.BestAR]
        fig2 = plt.figure(figsize=(15,15))
        ax1 = fig2.add_subplot(111)
        ax1.plot(range(0,len(Y)), Y,marker='', color='olive', linewidth=2)
        ax1.plot(range(0,len(Y_fit)), Y_fit, marker='', color='blue', linewidth=2)
        ax1.legend(['Fitted', 'data'], loc='upper left')
        plt.show()        
        



class OptimMonthly:
    def __init__(self, GDP, monthly):
        self.GDP = GDP
        self.monthly = monthly

        
    def Forecastperf(self, skip, maxlag):
        self.skip = skip
        self.maxlag = maxlag
        monthly = self.monthly[0:]
        length = np.int(np.ceil(len(monthly)/3))
        monthly = np.append(monthly, np.full(length*3-len(monthly),np.nan)) # fill remainder of monthly with NaN
        startq = np.int(np.ceil(maxlag/3)) # to match indexing convention
        GDP = self.GDP[startq:,0:] # allow space for lagged monthly data
        GDPlag = self.GDP[startq-1:,0:]
        X = np.zeros(shape=(length-startq,maxlag+1,3))
        # create various lag lengths
        # X time, lag, month (month 1, 2, 3), i.e. Jan, Feb, March in Q1
        ###########################
        # Make store of regressors X (quarter, monthly data up to max lag, month of quarter)
        for jj in range(0,3):
            for ii in range(0,maxlag+1):
                X[0:,ii,jj] = monthly[startq*3-ii+jj:len(monthly)-ii+jj:3].T # every third element for each lag 
         
        Y = GDP.reshape(-1,1)
        Ylag = GDPlag.reshape(-1,1)
        
        # create fitted values and test RMSE
        
        Fit_val = np.zeros(shape=(length-skip-startq,maxlag,3))
        RMSE = np.zeros(shape=(3,maxlag))
        Fit_valAR = np.zeros(shape=(length-skip-startq,maxlag,3))
        RMSEAR = np.zeros(shape=(3,maxlag))
        
        #Model = [[None for col in range(maxlag)] for row in range(3)] # holds latest monthxlag regression
        for pp in range(0,3): # Months
            for ii in range(1,maxlag+1): # Up to maxlag
                for jj in range(skip, len(X)): 
                    if jj<=len(Y):
                        # Exclude nan values if early series data not available
                        RegDatX, RegDatY = X[0:jj,0:ii,pp].reshape(-1,ii), Y[0:jj,0].reshape(-1,1)
                        Model = LinearRegression().fit(RegDatX[~np.isnan(RegDatX).any(axis=1),0:].reshape(-1,ii), RegDatY[~np.isnan(RegDatX).any(axis=1),0])
                        if np.isnan(X[jj:jj+1,0:ii,pp]).any():
                            Fit_val[jj-skip:jj+1-skip,ii-1,pp] = np.nan
                        else:
                            Fit_val[jj-skip:jj+1-skip,ii-1,pp] = Model.predict(X[jj:jj+1,0:ii,pp].reshape(-1,ii)).T
                            
                        RegDatXAR = np.concatenate((X[0:jj,0:ii,pp].reshape(-1,ii), Ylag[0:jj,0].reshape(-1,1)), axis=1)
                        ModelAR = LinearRegression().fit(RegDatXAR[~np.isnan(RegDatX).any(axis=1),0:].reshape(-1,ii+1), RegDatY[~np.isnan(RegDatXAR).any(axis=1),0])
                        if np.isnan(X[jj:jj+1,0:ii,pp]).any():
                            Fit_valAR[jj-skip:jj+1-skip,ii-1,pp] = np.nan
                        else:
                            Xpred = np.concatenate((X[jj:jj+1,0:ii,pp].reshape(-1,ii), Ylag[jj:jj+1,0].reshape(-1,1)), axis=1)
                            Fit_valAR[jj-skip:jj+1-skip,ii-1,pp] = ModelAR.predict(Xpred).T
                        
                RMSE[pp,ii-1] = np.sqrt(np.average(np.square(Y[skip:,0]-Fit_val[0:len(Y)-skip,ii-1,pp]))) # don't include no data but
                RMSEAR[pp,ii-1] = np.sqrt(np.average(np.square(Y[skip:,0]-Fit_valAR[0:len(Y)-skip,ii-1,pp]))) # don't include no data but
                
                BestnoAR = RMSE.argmin(axis=1)
                BestAR = RMSEAR.argmin(axis=1)
                self.BestAR = RMSE.argmin(axis=1)
                self.OptimRMSE = RMSE[[0,1,2], [self.BestAR[0],self.BestAR[1],self.BestAR[2]]]
                
        self.OptimFit = np.zeros(shape=(len(Fit_val),3))        
        for ii in range(0,3):
            if RMSE[ii, BestnoAR[ii]]< RMSEAR[ii, BestAR[ii]]:
                self.OptimFit[0:,ii] = Fit_val[:,BestnoAR[ii],ii]
                self.BestAR[ii] = BestnoAR[ii]
                self.OptimRMSE[ii] = RMSE[ii, BestnoAR[ii]]
            else:
                self.OptimFit[0:,ii] = Fit_valAR[:, BestAR[ii],ii]
                self.BestAR[ii] = BestnoAR[ii]
                self.OptimRMSE[ii] = RMSEAR[ii, BestAR[ii]]
        # find best lag for each month for AR and no AR - then choose best between AR and no AR.
        
            

        
    # def ForecastperfARIMA(self, skip, maxlag):
    #     self.skip = skip
    #     self.maxlag = maxlag
    #     monthly = self.monthly[0:]
    #     length = np.int(np.ceil(len(monthly)/3))
    #     monthly = np.append(monthly, np.full(length*3-len(monthly),np.nan)) # fill remainder of monthly with NaN
    #     startq = np.int(np.ceil(maxlag/3)) # to match indexing convention
    #     GDP = self.GDP[startq:,0:] # allow space for lagged monthly data
    #     X = np.zeros(shape=(length-startq,maxlag+1,3))
    #     # create various lag lengths
    #     # X time, lag, month (month 1, 2, 3), i.e. Jan, Feb, March in Q1
    #     ###########################
    #     # Make store of regressors X (quarter, monthly data up to max lag, month of quarter)
    #     for jj in range(0,3):
    #         for ii in range(0,maxlag+1):
    #             X[0:,ii,jj] = monthly[startq*3-ii+jj:len(monthly)-ii+jj:3].T # every third element for each lag 
         
    #     Y = GDP.reshape(-1,1)
    #     # create fitted values and test RMSE
        
    #     Fit_val = np.zeros(shape=(length-skip-startq,maxlag,3))
    #     RMSE = np.zeros(shape=(3,maxlag))
        
    #     #Model = [[None for col in range(maxlag)] for row in range(3)] # holds latest monthxlag regression
    #     for pp in range(0,3): # Months
    #         for ii in range(1,maxlag+1): # Up to maxlag
    #             for jj in range(skip, len(X)): 
    #                 if jj<=len(Y):
    #                     # Exclude nan values if early series data not available
    #                     RegDatX, RegDatY = X[0:jj,0:ii,pp].reshape(-1,ii), Y[0:jj,0].reshape(-1,1)
    #                     Model = ARIMA(RegDatY[~np.isnan(RegDatX).any(axis=1),0], order = (1,0,1), exog = RegDatX[~np.isnan(RegDatX).any(axis=1),0:].reshape(-1,ii)).fit()
    #                     if np.isnan(X[jj:jj+1,0:ii,pp]).any():
    #                         Fit_val[jj-skip:jj+1-skip,ii-1,pp] = np.nan
    #                     else:
    #                         Fit_val[jj-skip:jj+1-skip,ii-1,pp] = Model.predict(start = len(RegDatY[~np.isnan(RegDatX).any(axis=1),0]),
    #                                                                            end = len(RegDatY[~np.isnan(RegDatX).any(axis=1),0]),
    #                                                                            exog = X[jj:jj+1,0:ii,pp].reshape(-1,ii))
                        
    #             RMSE[pp,ii-1] = np.sqrt(np.average(np.square(Y[skip:,0]-Fit_val[0:len(Y)-skip,ii-1,pp]))) # don't include no data but
            
    #     self.RMSE = RMSE
    #     self.Fit_val = Fit_val
    #     self.BestAR = RMSE.argmin(axis=1) # location of lowest for each month
    #     self.OptimFit = np.zeros(shape=(len(Fit_val),3)) # Useful for later functions just to have the optimal lag structure in each month
    #     for ii in range(0,3):
    #         self.OptimFit[0:,ii] = Fit_val[:,self.BestAR[ii],ii]
    #     self.OptimRMSE = RMSE[[0,1,2], [self.BestAR[0],self.BestAR[1],self.BestAR[2]]] # store optimal RMSE minimums at each month
            
    def PlotBest(self):
        Y = self.GDP
        maxlag = self.maxlag 
        startq = np.int(np.ceil(maxlag/3))
        Y = Y[startq+self.skip:]
        Y_fit = self.Fit_val[0:,0:,0:]
        fig2 = plt.figure(figsize=(15,15))
        title = ['Month 1', 'Month 2', 'Month 3']
        for num, tt in zip(range(0,3), title):
            ax1 = fig2.add_subplot(3,1,num+1)
            ax1.plot(range(0,len(Y)), Y,marker='', color='olive', linewidth=2)
            ax1.plot(range(0,len(Y_fit)), Y_fit[0:,self.BestAR[num],num], marker='', color='blue', linewidth=2)
            ax1.legend(['GDP', 'fitted'], loc='upper left')
            ax1.set_title(tt)
        plt.show()      

        
class ForecastCombine:
    def __init__(self, GDP, monthlyseries, skip, ARt, maxlag , ARinclude, weighttype, names):
        self.GDP = GDP
        self.monthlyseries = monthlyseries
        self.skip = skip
        self.ARt = ARt
        self.maxlag = maxlag
        self.names = names
        self.ARinclude = ARinclude
        self.weighttype = weighttype
        
    def Optimize(self):
        # Get optimal AR structure
        GDPfitted = OptimARoos(GDP = self.GDP)
        GDPfitted.Forecastperf(ARt = self.ARt, skip = self.skip)
        ARfit = GDPfitted.OptimFit
        ARRMSE = GDPfitted.OptimRMSE
        ARlag = GDPfitted.BestAR
        if self.ARt>np.int(np.ceil(self.maxlag/3)):
            self.addiskip = self.ARt-np.int(np.ceil(self.maxlag/3)) # how many periods to skip in sample due to lags on AR1 and monthly
        else:
            self.addiskip = 0
        size = len(self.GDP)-(np.int(np.ceil(self.maxlag/3))+self.skip+self.addiskip)
        Month1 = np.zeros(shape=(size+1,len(self.monthlyseries))) # additional row for forecast
        Month2 = np.zeros(shape=(size+1,len(self.monthlyseries)))
        Month3 = np.zeros(shape=(size+1,len(self.monthlyseries)))
        Month1_RMSE = np.zeros(shape=(1,len(self.monthlyseries)))
        Month2_RMSE = np.zeros(shape=(1,len(self.monthlyseries)))
        Month3_RMSE = np.zeros(shape=(1,len(self.monthlyseries)))
        Month1lag = np.zeros(shape=(1,len(self.monthlyseries)))
        Month2lag = np.zeros(shape=(1,len(self.monthlyseries)))
        Month3lag = np.zeros(shape=(1,len(self.monthlyseries)))
        
        for series, jj in zip(self.monthlyseries, range(0, len(self.monthlyseries))):
            Temp = OptimMonthly(GDP = self.GDP[self.addiskip:], monthly = series[self.addiskip*3:])
            Temp.Forecastperf(skip=self.skip, maxlag = self.maxlag)
            Month1[0:,jj], Month2[0:,jj], Month3[0:,jj] = Temp.OptimFit[0:,0], Temp.OptimFit[0:,1], Temp.OptimFit[0:,2]
            Month1_RMSE[0,jj], Month2_RMSE[0,jj], Month3_RMSE[0,jj] = Temp.OptimRMSE[0], Temp.OptimRMSE[1], Temp.OptimRMSE[2]
            Month1lag[0,jj], Month2lag[0,jj], Month3lag[0,jj] = Temp.BestAR[0]+1, Temp.BestAR[1]+1, Temp.BestAR[2]+1
            
        # Add on AR forecast if needed
        if self.ARinclude:
            Month1, Month2, Month3  = np.append(Month1, ARfit.reshape(-1,1), axis=1), np.append(Month2, ARfit.reshape(-1,1), axis=1),np.append(Month3, ARfit.reshape(-1,1), axis=1)
            Month1_RMSE, Month2_RMSE, Month3_RMSE  = np.append(Month1_RMSE, ARRMSE.reshape(-1,1), axis=1), np.append(Month2_RMSE, ARRMSE.reshape(-1,1), axis=1),np.append(Month3_RMSE, ARRMSE.reshape(-1,1), axis=1)
            Month1lag, Month2lag, Month3lag  = np.append(Month1lag, ARlag.reshape((-1,1))+1, axis=1), np.append(Month2lag, ARlag.reshape((-1,1))+1, axis=1), np.append(Month3lag, ARlag.reshape((-1,1))+1, axis=1)
            self.names.append('AR')
        # Get RMSE_weighted forecast
        Optimal = np.zeros(shape=(size+1,3))
        # Use inverse of RMSE or MSE to weight together forecasts
        if self.weighttype == 'rmse':
            Optimal[0:,0], Optimal[0:,1], Optimal[0:,2] =  WeightedMeanNaN(Tseries = Month1, weights=np.divide(1,Month1_RMSE)), WeightedMeanNaN(Tseries = Month2, weights=np.divide(1,Month2_RMSE)), WeightedMeanNaN(Tseries = Month3, weights=np.divide(1,Month3_RMSE))
        elif self.weighttype == 'mse':
            Optimal[0:,0], Optimal[0:,1], Optimal[0:,2] =  WeightedMeanNaN(Tseries = Month1, weights=np.divide(1,np.square(Month1_RMSE))), WeightedMeanNaN(Tseries = Month2, weights=np.divide(1,np.square(Month2_RMSE))), WeightedMeanNaN(Tseries = Month3, weights=np.divide(1,np.square(Month3_RMSE)))
       
        RMSE = np.zeros(shape = (1,3))
        # Get newly calculated RMSE
        for ii in range(0,3):
            RMSE[0,ii] = np.sqrt(np.average(np.square(self.GDP[len(self.GDP)-size:,0]-Optimal[0:-1,ii])))
        
        self.RMSEcombined = RMSE
        self.size = size # size of forecast given max lag settings for monthly and ar1
        self.OptimalFit = Optimal
        self.Month1, self.Month2, self.Month3 = Month1, Month2, Month3
        self.Month1_RMSE, self.Month2_RMSE, self.Month3_RMSE = Month1_RMSE, Month2_RMSE, Month3_RMSE
        self.Month1lag, self.Month2lag, self.Month3lag = Month1lag, Month2lag, Month3lag
        
    def PlotBest(self, datetime, Quarterlyname):
        Y = self.GDP
        maxlag = self.maxlag 
        Y = Y[len(Y)-self.size:]
        Y_fit = self.OptimalFit
        fignew = plt.figure(figsize=(15,15))
        title = ['Month 1', 'Month 2', 'Month 3']
        for num, tt in zip(range(0,3), title):
            axnew = fignew.add_subplot(3,1,num+1)
            axnew.plot(datetime[-len(Y)-1:-1], Y,marker='o', color='olive', linewidth=2)
            axnew.plot(datetime[-len(Y_fit):], Y_fit[0:,num], marker='o', color='blue', linewidth=2)
            axnew.legend([Quarterlyname, 'fitted'], loc='upper left')
            axnew.set_title(tt)
        plt.show()   
        
        
    def PrintNiceOutput(self, datetime):
        # Print optimal fit figures
        Qoffcast = datetime[-1].strftime('%d-%b-%Y')
        titlefit = ['Month 1', 'Month 2', 'Month 3']
        optimfit = self.OptimalFit[-1].flatten().tolist()
        print('\n')
        print('Optimal forecast for quarter ending ' + Qoffcast, end='\n')
        print(tabulate(np.vstack((titlefit, optimfit))))
        optimrmses = self.RMSEcombined.flatten().tolist()
        print('\n')
        print('Optimal forecast RMSEs')
        print(tabulate(np.vstack((titlefit, optimrmses))))
        
        print('\n')
        print('Forecast of each indicator in each month')
        month1 = self.Month1[-1,:].flatten().tolist()
        month1.insert(0, "Month1")
        month2 = self.Month2[-1,:].flatten().tolist()
        month2.insert(0, "Month2")
        month3 = self.Month3[-1,:].flatten().tolist()
        month3.insert(0, "Month3")
        fcasttitle = self.names.copy()
        fcasttitle.insert(0, '')
        print(tabulate(np.vstack((fcasttitle, month1, month2, month3 )), headers = "firstrow"))

        print('\n')
        print('Out of sample RMSE for each indicator in each month')
        rmse1 = self.Month1_RMSE.flatten().tolist()
        rmse1.insert(0, "Month1")
        rmse2 = self.Month2_RMSE.flatten().tolist()
        rmse2.insert(0, "Month2")
        rmse3 = self.Month3_RMSE.flatten().tolist()
        rmse3.insert(0, "Month3")
        print(tabulate(np.vstack((fcasttitle, rmse1, rmse2, rmse3 )), headers = "firstrow"), end='\n')
        
        print('\n')
        print('Optimal number of lags for each indicator')
        lags1 = self.Month1lag.flatten().tolist()
        lags1.insert(0, "Month1")
        lags2 = self.Month2lag.flatten().tolist()
        lags2.insert(0, "Month2")
        lags3 = self.Month3lag.flatten().tolist()
        lags3.insert(0, "Month3")
        print(tabulate(np.vstack((fcasttitle, lags1, lags2, lags3 )), headers = "firstrow"), end='\n')
                                                          
                                                          

        
def WeightedMeanNaN(Tseries, weights):
    ## calculates weighted mean 
    N_Tseries = Tseries.copy()
    Weights = np.repeat(weights, len(N_Tseries), axis=0) # make a vector of weights matching size of time series
    loc = np.where(np.isnan(N_Tseries)) # get location of nans
    Weights[loc] = 0
    N_Tseries[loc] = 0
    
    locallmissing = np.where(Weights.sum(axis=1) == 0)
    Weights[locallmissing, :] = 1 # to stop divide by zero error
    Weights = Weights/Weights.sum(axis=1)[:,None] # normalize each row so that weights sum to 1
    WeightedAve = np.multiply(N_Tseries,Weights)
    WeightedAve = WeightedAve.sum(axis=1)
    WeightedAve[locallmissing] = np.nan 
    return WeightedAve
    
    
def DelaySeries(DFmonth, Delay):
    Delay2 = np.asarray(Delay)
    # add row if no space to lag series
    if np.any(~np.isnan(DFmonth.iloc[-1,Delay2==1])):
        last_date = DFmonth.iloc[[-1]].index+1
        DFmonth = DFmonth.append(pd.DataFrame(index=[last_date]))
    DFmonthhold = DFmonth.copy()
    DFmonth = DFmonth[1*3:] # Get rid of first three months to match Quarterly
    # Add extra row if shift goes into non-existent month
    
    
    for ii in range(len(Delay)):

        if Delay[ii]==0:
            DFmonth.iloc[:,ii]= DFmonthhold.iloc[1*3:,ii]
        else:
            DFmonth.iloc[:,ii] = DFmonthhold.iloc[1*3-Delay[ii]:-Delay[ii], ii].values  
        
    return DFmonth
    