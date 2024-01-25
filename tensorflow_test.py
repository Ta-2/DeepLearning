import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Input

datas = np.loadtxt("iris.csv")
#print(datas)
X = datas[:,0:4].astype("float64")
X_max, X_min = X.max(), X.min()
X = (X-X_min)/(X_max-X_min)
Y = datas[:,-1].astype("uint8")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=3)

activation_func = "linear"
model = Sequential([
    Input(shape=(4,)),
    #Dense(5, activation=activation_func, input_dim=4),
    #Dense(4, activation=activation_func),
    Dense(3, activation=activation_func),
    Dense(3, activation="softmax")
], name="my_model")
#model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, Y_train, 
    batch_size=5, epochs=200,
    validation_split=0.2, verbose=0
)

for i, w in enumerate(model.get_weights()):
    print(f"{i}layer's weight shape: {w.shape}")
    print(f"{i}layer's weights: {w}")

test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"activation function: {activation_func}")
print(f"loss: {test_loss}")
print(f"accuracy: {test_acc}")

predictions = model.predict(X_test)
results = predictions.argmax(axis=1)
print(results)
print(Y_test)