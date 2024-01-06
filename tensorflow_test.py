import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten

datas = np.loadtxt("iris.csv")
#print(datas)
X = datas[:,0:4].astype("float64")
X_max, X_min = X.max(), X.min()
X = (X-X_min)/(X_max-X_min)
#print(X[0:5])
Y = datas[:,-1].astype("uint8")
#print(X.shape)
#print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=3)
#print(X_train.shape)
#print(Y_train.shape)
#print(X_test.shape)
#print(Y_test.shape)

model = Sequential([
    Dense(3, activation="tanh", input_dim=4),
    #Dense(4, activation="sigmoid"),
    #Dense(3, activation="sigmoid"),
    Dense(3, activation="softmax")
], name="my_model")
model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, Y_train, 
    batch_size=5, epochs=200,
    validation_split=0.2#, verbose=0
)

for a in model.get_weights():
    print(a.shape)
    print(a)

test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"loss: {test_loss}")
print(f"accuracy: {test_acc}")

predictions = model.predict(X_test)
#print(predictions)
results = predictions.argmax(axis=1)
print(results)
print(Y_test)