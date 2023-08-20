# Work in progress

currently results are pretty bad will upload if they improve

training on https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis dataset
entity labels removed. Am only using the data from training.csv split into 3.
The content of the csv file have been extracted and split into the text.txt and label.txt so I can easily use them with fstream



UPDATE: yet another bug caught, wasn't storing pre activstionms correctly, was using .reserve on a vector instead of .resize , so when you try and access it it is out of index but at the same time it doesn't corrupt things
