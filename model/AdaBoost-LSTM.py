import torch
import torch.nn as nn
import pickle5 as pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable 
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#import seaborn as sns
import math
import matplotlib.pyplot as plt


dataframe = pd.read_csv("./dataset/full_preprocessed_data.csv", index_col="Timestamp")
dataframe = dataframe.loc[:, dataframe.columns != "total"]
dataframe = dataframe.iloc[:, :]

dataframe_v = dataframe.to_numpy()

window = 7

# Splitting the data
datasize_length = dataframe_v.shape[0]

# The data split percentages
training_percentage = 0.7
testing_percentage = 0.15
validation_percentage = 0.15

testing_15 = 0.15
# Number of samples in each split
num_train_samples = int(training_percentage * datasize_length)
num_test_samples = int(testing_percentage * datasize_length)
num_test_samples_new = int(testing_15 * datasize_length)
num_validation_samples = int(validation_percentage * datasize_length)

def create_sequences(x, window):
    newDataframe =[]
    for rowIndex in range(x.shape[0]-window):
        inputSequence = []
        newDataframe.append(x[rowIndex: rowIndex+window])
        #newDataframe.append(inputSequence)

    return np.array(newDataframe)

newDf = create_sequences(dataframe.to_numpy(), window)

x = dataframe.iloc[:-1, :]
#x = dataframe.iloc[:-1].loc[:,["hash_rate", "Block_size", "Difficulty", "active_addresses", "Block_time", "Average fees", "mining_profitability", "Transactions"]]
y = dataframe.iloc[1:, 1:2]


mm = MinMaxScaler()
ss = StandardScaler()

x_ss = ss.fit_transform(x)
y_mm = mm.fit_transform(y)


x_train = x_ss[:num_train_samples, :]
x_test = x_ss[num_train_samples:, :]

y_train = y_mm[:num_train_samples, :]
y_test = y_mm[num_train_samples:, :]

x_train = create_sequences(x_train, window)
y_train = y_train[:-window]

x_test = create_sequences(x_test, window)
y_test = y_test[:-window]

x_train_tensors = Variable(torch.Tensor(x_train))
x_test_tensors = Variable(torch.Tensor(x_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test)) 

x_train_tensors_final = x_train_tensors
x_test_tensors_final = x_test_tensors


num_epochs = 700 #1000 epochs
learning_rate = 0.00003 #0.001 lr

input_size = 18 #number of features
hidden_size = 512 #number of features in hidden state
num_layers = 3 #number of stacked lstm layers

num_classes = 1 #number of output classes 
device = "cpu"
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = x.to(device)
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = output[:, -1, :]
        out = self.relu(out)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, x_train_tensors_final.shape[1]) #our lstm class
lstm1 = lstm1.to(device)
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)
df = pd.read_csv("./dataset/full_preprocessed_data.csv")
X = df.iloc[:-1, 1:].to_numpy()
y = df.iloc[1:, 2:3].to_numpy().flatten()
train_per = 0.7
test_per = 0.15
val_per = 0.15

train_split =  int(train_per * X.shape[0])
test_split = int(test_per * X.shape[0])

X_train = X[test_split+test_split:]
X_test = X[test_split:test_split+test_split]
X_val = X[:test_split]

y_train = y[test_split+test_split:]
y_test = y[test_split:test_split+test_split]
y_val = y[:test_split]

minLoss = np.inf
minEpoch = 0

how_many_to_stop = 100
last = 0


for m in range(4):
    for epoch in range(num_epochs):
        
        outputs = lstm1.forward(x_train_tensors_final) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0
        # obtain the loss function
        adaBoostLstm = DecisionTreeRegressor(random_state=0, max_depth=20)
        adaBoostLstm = AdaBoostRegressor(base_estimator=adaBoostLstm, random_state=0, n_estimators=80)
        adaBoostLstm.fit(X_train, y_train)

        y_train_tensors = y_train_tensors.to(device)
        loss = criterion(outputs, y_train_tensors)

        loss.backward() #calculates the loss of the loss function

        optimizer.step() #improve from loss, i.e backprop
        
        with torch.no_grad():
            lstm1.eval()
            x_test_tensors_final=x_test_tensors_final.to(device)
            y_test_tensors=y_test_tensors.to(device)
            outputs = lstm1(x_test_tensors_final)
            test_loss = criterion(outputs, y_test_tensors)
            if (test_loss < minLoss):
                print("LOSS DECREASE ========> From: {0}, To: {1}".format(minLoss, test_loss))
                
                minLoss = test_loss
                minEpoch = epoch
                last=0
            last+=1
            lstm1.train()
        if epoch % 50 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
        if last > how_many_to_stop:
            out_test = adaBoostLstm.predict(X_test)
            out_val = adaBoostLstm.predict(X_val)
            out_train = adaBoostLstm.predict(X_train)
            mae_test = mean_absolute_error(y_test, out_test)
            mae_train = mean_absolute_error(y_train, out_train)
            mae_val = mean_absolute_error(y_val, out_val)

            mse_test = mean_squared_error(y_test, out_test)
            mse_train = mean_squared_error(y_train, out_train)
            mse_val = mean_squared_error(y_val, out_val)
            print('''
                Training stopped, validation loss is not decreasing anymore!
                
                RMSE Train: {6},
                MSE Train: {3},
                MAE Train: {0},
                RMSE Validation: {7},
                MSE Validation: {4},
                MAE Validation: {1},
                RMSE Test: {8},

                MSE Test: {5},
                MAE Test: {2},
                '''.format(mae_train, mae_val, mae_test, mse_train, mse_val, mse_test, math.sqrt(mse_train), math.sqrt(mse_val), math.sqrt(mse_test), epoch))
            break

