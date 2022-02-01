import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#import seaborn as sns
import matplotlib.pyplot as plt
import pickle5 as pickle
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--date', '-d', help="Select the supposed to be the current date", type= str)
args = parser.parse_args()
#print(parser.format_help())
from utils.utils import getPricePrediction

exec(open("./model/AdaBoost-LSTM.py").read())
getPricePrediction(args.date)
