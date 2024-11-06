from numpy import loadtxt
from keras import models

# Load the entire model directly
model = models.load_model("model.h5") # used to load the entire model directory
print("Loaded model from the disk")

# Load dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",") # loading the dataset
x = dataset[:, 0:8] # It selects the columns from 0 to 8
y = dataset[:, 8] # The output is the final column

# Make predictions
predictions = model.predict(x)# The model is used to predict the output
for i in range(10, 15): # The range is from 10 to 15 ,We have only selected these rows
    print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))# The model is used to predict the output and display the results
