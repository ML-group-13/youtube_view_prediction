# Machine learning: Youtube view prediction
### University of Groningen, Artificial Intelligence, Machine Learning
Jussi Boersma (s2779153), Ryanne Hauptmeijer (s2984326), Xabi Krant (s2955156), Wessel van der Rest (s2873672).

We want to do something with predicting the number of views based on the data here, but also on features extracted from the thumbnails.

Dataset: https://www.kaggle.com/datasnaek/youtube-new#USvideos.csv

To run program:
```
$python pipe.py
```

It is also possible to run in a development mode, this will use 10 000 data points by default or the amount used as input: 
```
$python pipe.py development
$python pipe.py development 1000
```

To save images from thumbnails, this does not need to be done when the data is already present:
```
$python save_images.py
```

To install all required packages:
```
$pip install xgboost
$pip install scikit-learn
$pip install numpy
$pip install pandas
$pip install matplotlib
$pip install opencv-python
```
