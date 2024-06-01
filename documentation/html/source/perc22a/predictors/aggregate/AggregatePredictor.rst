===================
AggregatePredictor
===================

What is AggregatePredictor?
---------------------------
AggregatePredictor is a class that can be used to combine multiple cone predictions
into a single prediction. Our Perceptions library has a number of cone predictors.
It has a lidar predictor and 2 predictor that calculate cone position relative
to their sensors. AggregatePredictor can be used to combine these predictions into
a single prediction.