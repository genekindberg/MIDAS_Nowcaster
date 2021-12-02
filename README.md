# MIDAS_Nowcaster
A tool to nowcast quarterly data with monthly indicators: US consumption example 

Pulls data directly from FRED from a list of codes - any target quarterly variable or monthly high frequency indicator can be used. Also works with Haver if available.

Calculates the optimal lag length to minimize out-of-sample root mean squared error (RMSE) for each indicator and whether the specification improves with an AR term included. Changes the specification for the optimal lag length in each month of the quarter.

Optimally combines the predicted nowcast using either the mean squared error or the root mean squared error to weight together each indicator. 

Displays the nowcast for each indicator and the optimally combined set of indicators. Displays the RMSE of each indicator and the RMSE of the optimally-combined nowcast. This allows the user to apply judgement if they believe a particular indicator is giving a poor steer in that quarter.

Currently set up to nowcast US consumption using monthly PCE data, non-farm payrolls, and real retail sales. Other indicators or a different target variable (such as GDP) can be easily switched by changing the FRED code.

