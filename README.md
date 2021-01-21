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

#Clean and Preprocessed Data Link
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

#Reading and shuffling of data
        df_promoter = pd.read_csv("read file",names = ['Y', 'X'])
        df_promoter_shuffle = pd.read_csv("read file",names = ['Y', 'X'])
        df_promoter_shuffle.drop(df_promoter_shuffle.index[0])
        df_promoter_shuffle.reset_index(inplace = True)
        df_promoter_shuffle.drop(['index'], axis = 1, inplace= True)
        df_promoter_shuffle['Y'] = 'Shuffled Promoters'

        def shuffler(sequence, k_let):
            length = [sequence[i:i+k_let] for i in range(0,len(sequence),k_let)]
            np.random.shuffle(length)
            return ''.join(length)

        k_let = #
        df_shuf = df_promoter_shuffle['X'].apply(shuffler,args = [k_let])
        df_shuf = pd.concat([df_promoter_shuffle['Y'], df_shuf],axis=1)
        df_shuf.head()
        
#Drop Duplicates: 

        labels = df['Y']
        X = df['X']
        X_drop_dup = X.drop_duplicates()
        idx = X_drop_dup.index
        data = np.array(df)
        df_final = pd.DataFrame(data[idx])
        df_final.columns = ['Y','X']
        df_final['Y'].value_counts().plot('bar',color = 'orange')

#Choosing Training sample count:
        limit = 35000
        df_final_promoter= df_final[df_final['Y'] == 'Fungus_Promoter'][:limit]
        df_final_shuf=df_final[df_final['Y'] == 'Shuffled Promoters'][:limit]
        df_final = pd.concat([df_final_promoter,df_final_shuf],ignore_index=True)
        df_final.tail() # New dataframe with recurring samples eliminated
        df_final.head()
        
 #k-mer creation       
        def getKmers(X, size=4):
                return [X[x:x+size].lower() for x in range(len(X) - size + 1)]
        df_final['words']=df_final.apply(lambda x: getKmers(x['X']), axis=1)
        df_final.drop('X',axis=1, inplace= True)
      
 #CNN architecture:
        
        model = Sequential()
        model.add(Embedding(vocab_size,128,input_length=max_len))
        model.add(Conv1D(filters=128, kernel_size=5,padding='same'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Conv1D(filters=64, kernel_size=5, padding='same'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Conv1D(filters=32, kernel_size=5, padding='same'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

#LSTM architecture:

        model = Sequential()
        model.add(Embedding(vocab_size, 50 ,input_length=max_len))
        model.add(LSTM(128))
        model.add(Dense(64, activation=’relu’))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation=’sigmoid’))
    
       
#For binary classification:     
         
        Loss function: binary_crossentropy,
        Optimizer: adam
        number of epochs: 10

#For multispecies classification: 

        Loss function: sparse_categorical_crossentropy
        Optimizer: adam
        number of epochs: 10
        
