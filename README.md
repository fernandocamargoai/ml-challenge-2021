# MeLi Challenge 2021

This is the source code I used to compete on the Mercado Libre Challenge 2021 (https://ml-challenge.mercadolibre.com).

I adopted the strategy to create a probabilistic demand forecaster and use its probabilitic output to run "simulations" to get the out-of-stock predictions. My final submission was made with an ensemble of:
- Two DeepAR models (one forecasting the minutes_active variable and one that takes this variable as input to forecast demand) 
- One model called CausalDeepAR (that simultaneously forecast minutes_active and use it to forecast demand)

When running the models, I got the out-of-stock predictions for each simulation. I tried many ways to convert this sample of predictions into the expected probabilities, but none of them could beat simply using tweedie with the mean of the sample.
