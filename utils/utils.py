import numpy as np
import pickle5 as pickle

def create_sequences(x, window):
    newDataframe =[]
    for rowIndex in range(x.shape[0]-window):
        inputSequence = []
        newDataframe.append(x[rowIndex: rowIndex+window])
        #newDataframe.append(inputSequence)

    return np.array(newDataframe)


def getPricePrediction(date):
    
    with open('./outputs/dates.pkl', 'rb') as f:
        dates = pickle.load(f)
    with open("./outputs/adaboostlstmall.pkl", 'rb') as f:
        adaboostlstmall = pickle.load(f)
    with open('./outputs/labels.pkl', 'rb') as f:
        real = pickle.load(f)
    dateIndexes = np.where(dates == date)

    if (len(dateIndexes) == 0) :
        print("Date is in a wrong format or the date selected is missing from the given preprocessed dataset")

    else:
        index = dateIndexes[0] + 1
        predictedPrice = adaboostlstmall[index]
        realPrice = real[index][0][0]
        yesterdayPrice = real[index-1][0][0]

        print("The price prediction for the next day ({0}) is: {1}, while the real price is {2}".format(str(dates[index]), str(predictedPrice), str(realPrice)))