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

#Code snippets to develop a framework

## import files
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

## 
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

        k_let = 1
        df_terminator = df_promoter_shuffle['X'].apply(shuffler,args = [k_let])
        df_terminator = pd.concat([df_promoter_shuffle['Y'], df_terminator],axis=1)
        df_terminator.head()

        df = pd.concat([df_promoter,df_terminator],axis=0)
        df = df[df['X'].str.len() == 1000]
        df.shape
        
        print(df['X'].str.len().value_counts())  # All the sequences have equal length
        df.reset_index(inplace = True)
        df.drop(['index'], axis = 1, inplace= True)
        df.tail()
        df.head()
        
        df['Y'].value_counts().plot('bar',color = 'turquoise')
        df.describe()

##Drop Duplicates: 

        labels = df['Y']
        X = df['X']
        X_drop_dup = X.drop_duplicates()
        idx = X_drop_dup.index
        data = np.array(df)
        df_final = pd.DataFrame(data[idx])
        df_final.columns = ['Y','X']
        df_final['Y'].value_counts().plot('bar',color = 'orange')

##Choosing Training sample count:
   
        limit = 35000
        df_final_promoter= df_final[df_final['Y'] == 'Fungus_Promoter'][:limit]
        df_final_terminator=df_final[df_final['Y'] == 'Shuffled Promoters'][:limit]
        df_final = pd.concat([df_final_promoter,df_final_terminator],ignore_index=True)
        df_final.tail() # New dataframe with recurring samples eliminated
        df_final.head()
        
        df_final.reset_index(inplace = True)
        df_final.drop(['index'], axis = 1, inplace= True)
        df_final['Y'].value_counts().plot('bar',color = 'orange')
        
        def getKmers(X, size=4):
                return [X[x:x+size].lower() for x in range(len(X) - size + 1)]
        df_final['words']=df_final.apply(lambda x: getKmers(x['X']), axis=1)
        df_final.drop('X',axis=1, inplace= True)
        
        df_final.head()
        
        Y = df_final.Y
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        df_texts = list(df_final['words'])
        for item in range(len(df_texts)):
            df_texts[item] = ' '.join(df_texts[item])
        X=df_texts
        max_len = 997
        tok = Tokenizer(num_words=None)
        tok.fit_on_texts(X)
        vocab_size = len(tok.word_index) + 1
        sequences = tok.texts_to_sequences(X)
        sequences_matrix = sequence.pad_sequences(sequences,maxlen=None)
        
        print(sequences_matrix.shape)
        le.inverse_transform([0,1])

        X_train,X_test,Y_train,Y_test = train_test_split(sequences_matrix,Y,test_size=0.10)

        
##CNN architecture:
        
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

##LSTM architecture:

        model = Sequential()
        model.add(Embedding(vocab_size, 50 ,input_length=max_len))
        model.add(LSTM(128))
        model.add(Dense(64, activation=’relu’))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation=’sigmoid’))
    
##For classification: 
        model.summary()
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        import time
        start_time=time.time()
        print("Training Time Starts")
        model.fit(X_train,Y_train,batch_size=128,epochs=10,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
        xx=(time.time()) - (start_time)
        print(xx)

        y_pred1 = model.predict(X_test)
        confuse = confusion_matrix(Y_test, y_pred1.round())

        print(confuse)

        TP=confuse[1][1]
        FP=confuse[0][1]
        FN=confuse[1][0]
        TN=confuse[0][0]

        accuracy= (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F=2*(precision * recall)/(precision + recall)
        specificity= TN / (TN + FP) 
        Error_Rate= (FP + FN)/(TP + FP + TN + FN)
        CC=((TP*TN)-(FP*FN))/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))

        print("accuracy {:0.2f}".format(accuracy))
        print("Precision {:0.2f}".format(precision))
        print("Recall/Sensitivity {:0.2f}".format(recall))
        print("F1 Score {:0.2f}".format(F))
        print("Specificity {:0.2f}".format(specificity))
        print("Error_Rate {:0.2f}".format(Error_Rate))
        print("correlation Coefficient {:0.2f}".format(CC))
   
##For binary classification:     
         
        Loss function: binary_crossentropy,
        Optimizer: adam
        number of epochs: 10

##For multispecies classification: 

        Loss function: sparse_categorical_crossentropy
        Optimizer: adam
        number of epochs: 10
