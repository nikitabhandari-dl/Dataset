**Software Requirements**

python 3

python packages: pandas, numpy, math, keras, sklearn, matplotlib

jupyter notebook


**Steps to follow:**

The project goal is to develop a framework to predict if a biological sequence belongs to a promoter or non-promoter category. Steps are as follows:

1. Remove duplicate sequences from promoter data.
2. Generate shuffle data from promoter sequences.
3. Perform k-merization.
4. Frequency based tokenization
5. Building and training classification algorithms:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RandomForestClassifier (from sklearn.ensemble)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;LSTM (from keras.layer)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CNN (from keras.layers.convolutional)
6. Performance evaluation


<br /><br /><br />
**Code snippets to develop a framework :**

**Drop Duplicates:**

X\_drop\_dup = X.drop\_duplicates()

idx = X\_drop\_dup.index

data = np.array(df)

df= pd.DataFrame(data[idx])

**Shuffle Data Snippet:**

def shuffler(sequence, k\_let):

length = [sequence[i:i+k\_let] for i in range(0,len(sequence),k\_let)]

np.random.shuffle(length)

return &#39;&#39;.join(length)

**K-merization:**

k=2 #K-mer size 2, 4 or8

def getKmers(X, size=k):

return [X[x:x+size].lower() for x in range(len(X) - size + 1)]

**CNN architecture:**

model = Sequential()

model.add(Embedding(vocab\_size,128,input\_length=max\_len))

model.add(Conv1D(filters=128, kernel\_size=5,padding=&#39;same&#39;))

model.add(MaxPooling1D(pool\_size=4))

model.add(Conv1D(filters=64, kernel\_size=5, padding=&#39;same&#39;))

model.add(MaxPooling1D(pool\_size=4))

model.add(Conv1D(filters=32, kernel\_size=5, padding=&#39;same&#39;))

model.add(MaxPooling1D(pool\_size=4))

model.add(Dense(1024, activation=&#39;relu&#39;))

model.add(Dropout(0.2))

model.add(Dense(512, activation=&#39;relu&#39;))

model.add(Dropout(0.2))

model.add(Dense(128, activation=&#39;relu&#39;))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(1, activation=&#39;sigmoid&#39;))

**LSTM architecture:**

model = Sequential()

model.add(Embedding(vocab\_size, 50 ,input\_length=max\_len))

model.add(LSTM(128))

model.add(Dense(64, activation=&#39;relu&#39;))

model.add(Dropout(0.5))

model.add(Dense(1, activation=&#39;sigmoid&#39;))

**For binary classification:**

Loss function: binary\_crossentropy,

Optimizer: adam

number of epochs: 10

**For multispecies classification:**

Loss function: sparse\_categorical\_crossentropy

Optimizer: adam

number of epochs: 10
