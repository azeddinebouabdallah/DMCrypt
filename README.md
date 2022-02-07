# Multimodal Approach for Cryptocurrency Price Prediction 

## Abstract 
Despite the recent improvements in cryptocurrency price predictions accuracy in
the last few years, this field is still far from an off-the-shelf topic. Unlike traditional
markets, cryptocurrency market prices are directly affected by several factors. Some
of the influences are the correlation between cryptocurrency and the global market,
public awareness, and the hash rate. This thesis proposes DMCrypt, a multimodal
AdaBoost-LSTM ensemble learning approach to tackle the problem of cryptocur-
rency price predictions using all the modalities driving the price fluctuations like
social media sentiments, search volumes, blockchain information, and trading data.
Experiment results show a promising improvement in price predictions over other
state-of-the-art approaches with an average RMSE decrease of $38 (19.29% improve-
ment). Extensive experiments further demonstrate the importance of each multi-
modality to the overall performance of the model, such that adding the blockchain
data or social media sentiments to the model decrease the prediction errors signifi-
cantly. Moreover, as an addition to the single price prediction, DMCrypt estimates
the distribution of the predicted price to allow modeling the uncertainty of such pre-
diction and provide better help for decision-making. To the best of my knowledge, 
this approach can be considered the first to combine all the multimodalities influ-
encing cryptocurrency prices and proposes an Adaboost-LSTM ensemble learning
architecture to be used in such a topic.

## Running the model
To run the model execute the following line:
```
python3 main.py --date 2020-02-10
```

__Note: the preprocessed dataset provided in this repository ranges from 2016-10-21 until 2021-03-08_