#Software Requirements
python 3
python packages: pandas, numpy, math, keras, sklearn, matplotlib 
jupyter notebook


#Steps to follow:

        The project goal is to develop a framework to predict if a biological sequence belongs to a 
        promoter or non-promoter category. Steps are as follows:
        1.	Remove duplicate sequences from promoter data.
        2.	Generate shuffle data from promoter sequences.
        3.	Perform k-merization.
        4.	Frequency based tokenization
        5.	Building and training classification algorithms:
        RandomForestClassifier (from sklearn.ensemble)
        LSTM (from keras.layer)
        CNN (from keras.layers.convolutional)
        6.	Performance evaluation  

#Dataset Link
https://drive.google.com/drive/folders/1VQ4r2SHGMmFMAq52oDhqYWz7cpTboeHE?usp=sharing


#Code snippets
#import files

        import pandas as pd
        import numpy as np
        from numpy.random import shuffle
        import math  
        from keras.preprocessing.text import Tokenizer
        from sklearn.preprocessing import LabelEncoder
        from keras.preprocessing import sequence
        from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import accuracy_score,  confusion_matrix, classification_report, roc_auc_score, roc_curve
        from sklearn.model_selection import train_test_split
        import matplotlib.pyplot as plt
        %matplotlib inline
        from sklearn.metrics import recall_score
        from sklearn.metrics import roc_auc_score, roc_curve
        from sklearn.preprocessing import OneHotEncoder
        from keras.models import Sequential
        from keras.layers.convolutional import Conv1D, MaxPooling1D
        from keras.layers import Dense, Dropout, Activation, Flatten
        from keras.callbacks import EarlyStopping
        import tensorflow as tf

#Shuffling of data
    
        def shuffler(sequence, k_let):
            length = [sequence[i:i+k_let] for i in range(0,len(sequence),k_let)]
            np.random.shuffle(length)
            return ''.join(length)
        k_let = #
                
 #k-mer creation
 
        def getKmers(X, size=4):
                return [X[x:x+size].lower() for x in range(len(X) - size + 1)]
        df_final['words']=df_final.apply(lambda x: getKmers(x['X']), axis=1)
        df_final.drop('X',axis=1, inplace= True)
      
       
#For binary classification:     
         
        Loss function: binary_crossentropy,
        Optimizer: adam
        number of epochs: 10

#For multispecies classification: 

        Loss function: sparse_categorical_crossentropy
        Optimizer: adam
        number of epochs: 10
        
