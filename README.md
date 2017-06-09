# time-series-lstm-experiments

Motivated by [this tutorial](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/).

Using LSTM(s) to predict the number of international airline passengers in units of 1,000. 

The dataset contains only 144 observations, maybe that is why going deeper or introducing more neurons (memory blocks) does not change the results. In fact, when using multiple, recent time steps to make the prediction for the next time step leads to worse results.
