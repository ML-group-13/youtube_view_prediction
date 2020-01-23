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

To save hd images from thumbnails run the save_images file. It is also possible to only save a number of images or save non-hd versions. The image features will not perform well without hd images. Images only need to be saved when the data is not already present:
```
$python save_images.py
$python save_images.py 3000
$python save_images.py non-hd
```

To install all required packages:
```
$pip install xgboost
$pip install scikit-learn
$pip install numpy
$pip install pandas
$pip install matplotlib
$pip install opencv-python
$pip install nltk
$pip install keras
$pip install tensorflow
```
