# time-series-lstm-experiments

Motivated by [this tutorial](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/).

Using LSTM(s) to predict the number of international airline passengers in units of 1,000. 

## Going deeper has no effect

The dataset contains only 144 observations, maybe that is why going deeper or introducing more neurons (memory blocks) does not really affect the results. In fact, when using multiple, recent time steps to make the prediction for the next time step leads to even worse results, which is strange.

## Normalization
### Concern
It seems like the default input normalization goes as follows. Consider input vector `x`, find `max(x), min(x)` and rescale `x` by mapping `min(x)->0`, `max(x)->1`. However if new input has bigger/smaller than previous `max(x)/min(x)` value the default normalization scheme would still map this value to `1/0`, which seems to be a problem. 
### Resolution
One way to resolve this problem is to assume that our input `x` has values from `0` to `infinity` and transform the data by mapping `0->1`, `infinity->0`. It can be done, for example, by using the function `y=1/(1+1/a exp[b x])`, where `a, b` are parameters and `x` is input as usual. Next we fix `a, b` by specifying the desired separation between `min(x)` and `max(x)`, let's say we choose `y(min(x))=0.01, y(max(x))=7/8`. Using this kind of transformation reduces the gap between training and testing results, which is the goal of any machine learning problem. 

Default scaler ![Default scaler](https://github.com/g3n1uss/time-series-lstm-experiments/blob/master/pics/LearningCurveDefaultScaler.png)

Improved scaler ![Improved scaler](https://github.com/g3n1uss/time-series-lstm-experiments/blob/master/pics/LearningCurveMyScaler.png)

## TODO:
Fix train/test data separation
