from numpy import loadtxt
from keras import models,layers,Sequential

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",") # We are loading the dataset
print(dataset) #The loaded dataset is printed
x=dataset[:,0:8] # It is the input part for the datset 0 is the starting point and goes till 8
y=dataset[:,8] # It is the output for thye datset ,It stores the 8 th column
print("input",x) #print only the input 
print("output",y) # print only the output
 #Adding layers

model=Sequential() # We are creating a sequential model
model.add(layers.Dense(12 , input_dim = 8 , activation = 'relu')) #Adding the first hidden layer
model.add(layers.Dense(8 , activation = 'relu')) #Adding the second hidden layer
model.add(layers.Dense(1 , activation = 'sigmoid'))# finla layer which determines the number of nurons in the output we use the sigmoid function
model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy']) #compiling the model ,Since we use the Sigmoid function we use binary_crossentropy

#Model Traing
model.fit(x,y ,epochs=50 ,batch_size=20) # we train the model for 20 times by sending the data 10 then next 10 then so on 
#Evaluation 
_,accuracy = model.evaluate(x,y) # this line is used to evaluate the model and return the loss and accuracy
print('Accuracy :%.2f'%(accuracy*100))# To show rthe accuracy in percentage

# Save the model 
model_jason = model.to_json() #The to_json() method returns a JSON string representation of the model
with open ("model.json","w") as json_file: #The open() method returs a file object
    json_file.write(model_jason) #The write() method writes the specified text to the file
model.save("model.h5") #The save() method saves the entire model (including architecture, optimizer state, and weights) and can have a filename ending with .h5 or .keras
print("The model is saved into the disk") # The model is saved into the disk