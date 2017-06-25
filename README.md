# time-series-lstm-experiments

Motivated by [this tutorial](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/).

Using LSTM(s) to predict the number of international airline passengers in units of 1,000. 

## Going deeper has no effect

The dataset contains only 144 observations, maybe that is why going deeper or introducing more neurons (memory blocks) does not really affect the results. Another possibility is that the entire dataset can be pretty well approximated by `sin(x) + x` function, which has only few parameters; using 1 LSTM has 12 parameters, which is enough to approximate that function. Another strange result is that when using multiple, recent time steps to make the prediction for the next time step leads to even worse results.

## Normalization
### Concern
It seems like the default input normalization goes as follows. Consider input vector `x`, find `max(x), min(x)` and rescale `x` by mapping `min(x)->0`, `max(x)->1`. However if new input has bigger/smaller than previous `max(x)/min(x)` value the default normalization scheme would still map this value to `1/0`, which seems to be a problem. 
### Resolution
One way to resolve this problem is to assume that our input `x` has values from `0` to `infinity` and transform the data by mapping `0->1`, `infinity->0`. It can be done, for example, by using the function `y=1/(1+1/a exp[b x])`, where `a, b` are parameters and `x` is input as usual. Next we fix `a, b` by specifying the desired separation between `min(x)` and `max(x)`, let's say we choose `y(min(x))=0.01, y(max(x))=7/8` (it turns out the choice of parameters is crucial for reducing the gap between train and validation errors, choosing 0.01 and 0.99 leads to bad results). Using this kind of transformation reduces the gap between training and testing results, which is the goal of any machine learning problem. 

Default scaler (min-max): 
![Default scaler](https://github.com/g3n1uss/time-series-lstm-experiments/blob/master/pics/LearningCurveDefaultScaler.png)

Improved (?) scaler: 
![Improved scaler](https://github.com/g3n1uss/time-series-lstm-experiments/blob/master/pics/LearningCurveMyScaler.png)

The best (?) scaler (tanh): 

![Tanh scaler](https://github.com/g3n1uss/time-series-lstm-experiments/blob/master/pics/LearningCurveTanhScaler.png)

Some (strange) results.

`testTrainSplit = 0.67, validationSplit = 0.1, num_epochs = 50`. 

|  Whole dataset for training | no | yes |
| --- | --- | ---|
|  **Scaler**         | **default/mine** | **default/mine** | 
| Train loss | 0.0017/0.0039 | 0.0033/0.0042 |
| Validation loss | 0.0045/0.0087 | 0.0107/0.0016 |
| Train score for normalized data | 0.04/0.07 | 0.05/0.06 |
| Test score for normalized data | 0.09/0.11 | 0.09/0.06 |
| Train score for non-normalized data | 22.80/24.73 | 24.06/23.22 |
| Test score for non-normalized data | 48.67/91.80 | 46.81/51.78 |

Finally let's plot data/predictions for the tanh scaler.

![](https://github.com/g3n1uss/time-series-lstm-experiments/blob/master/pics/TanhScaler.png)

## TODO:
Fix train/test data split
