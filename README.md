Software Requirements
python 3
python packages: pandas, numpy, math, keras, sklearn, matplotlib 
jupyter notebook
Datasets
link: 
Steps to follow:
The project goal is to develop a framework to predict if a biological sequence belongs to a       promoter or non-promoter category. Steps are as follows:
1.	Remove duplicate sequences from promoter data.
2.	Generate shuffle data from promoter sequences.
3.	Perform k-merization.
4.	Frequency based tokenization
5.	Building and training classification algorithms:
RandomForestClassifier (from sklearn.ensemble)
LSTM (from keras.layer)
CNN (from keras.layers.convolutional)
6.	Performance evaluation  

Code snippets to develop a framework
Drop Duplicates: 
X_drop_dup = X.drop_duplicates()
idx = X_drop_dup.index
data = np.array(df)
df= pd.DataFrame(data[idx])

Shuffle Data Snippet:
def shuffler(sequence, k_let):
    length = [sequence[i:i+k_let] for i in range(0,len(sequence),k_let)]
    np.random.shuffle(length)
    return ''.join(length)

K-merization: 
k=2    #K-mer size 2, 4 or8
def getKmers(X, size=k):
    return [X[x:x+size].lower() for x in range(len(X) - size + 1)]

CNN architecture:
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

LSTM architecture:
model = Sequential()
model.add(Embedding(vocab_size, 50 ,input_length=max_len))
model.add(LSTM(128))
model.add(Dense(64, activation=’relu’))
model.add(Dropout(0.5))
model.add(Dense(1, activation=’sigmoid’))
    
For binary classification: 
Loss function: binary_crossentropy,
Optimizer: adam
number of epochs: 10

For multispecies classification: 
Loss function: sparse_categorical_crossentropy
Optimizer: adam
number of epochs: 10

