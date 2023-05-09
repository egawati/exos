# EXOS : Explaining Outliers in Data Streams

Detecting outliers in real-time data streams is an important field of study. But currently, there isn't much research on how to explain those outliers. To figure out what's going on with these unusual data points, experts have to take a closer look - which can be tough when dealing with massive amounts of complex information. That's why presenting explanations along with the outliers themselves could speed things up!

An outlier explanation can be outlying attributes, a subset of features responsible for the outlier abnormality. Some techniques have been proposed to generate outlying attributes; however, none simultaneously addresses data streams' unbounded volume, concept drift, and cross-correlation characteristics. 

To fill this gap, we propose EXOS, a framework to generate outlying attributes of each outlier detected in multiple source data streams. It applies a single-pass incremental eigendecomposition model to capture data attribute correlation within each source and across sources. The single-pass approach allows EXOS to handle the unbounded volume of data streams. To adapt to concept drift, the model is updated periodically.
