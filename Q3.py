import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
class Featuredata(data.Dataset):
  def __init__(self, file):
    df=pd.read_csv(file,header=None)
    x=df.iloc[:,1:]
    y=df.iloc[:,0]
    sc=StandardScaler()
    x=sc.fit_transform(x)
    self.X_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=y
  def __len__(self):
    return len(self.y_train)
  def __getitem__(self,index):
    return self.X_train[index],self.y_train[index]
traindata=Featuredata('/content/drive/MyDrive/ML DataSet/largeTrain.csv')
valdata=Featuredata('/content/drive/MyDrive/ML DataSet/largeValidation.csv')
trainloader = data.DataLoader(traindata, batch_size=64, shuffle=True)
valloader=data.DataLoader(valdata,batch_size=64, shuffle=True)
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

hidden_size=[5,20,50,100,200] #given hidden layer sizes

criterion = nn.CrossEntropyLoss()
loss_size=np.zeros(len(hidden_size))
loss_val=np.zeros(len(hidden_size))
for s in range(len(hidden_size)):
  n=hidden_size[s]
  model=nn.Sequential(nn.Linear(128,n),nn.ReLU(),nn.Linear(n,10),nn.Softmax(dim=1))
  model.apply(init_weights)
  optimizer = optim.SGD(model.parameters(), lr=0.01)
  epochs=100
  loss_n=0.0
  for e in range(epochs):
    running_loss=0.0
    for x,y in trainloader:
      optimizer.zero_grad()
      output=model(x)
      loss=criterion(output,y)
      loss.backward()
      optimizer.step()
      running_loss+=loss.item()
    loss_n+=running_loss/len(trainloader)
  loss_n/=epochs
  loss_size[s]=loss_n
  valloss=0.0
  with torch.no_grad():
    for x,y in valloader:
      output=model(x)
      loss=criterion(output,y)
      valloss+=loss
  loss_val[s]=valloss/len(valloader)

  
      

  





  

lrates=[0.1,0.01,0.001]
for s in range(len(lrates)):
  plt.figure(figsize=(10,7))
  
  criterion = nn.CrossEntropyLoss()
  model=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,10),nn.Softmax())
  model.apply(init_weights)
  optimizer = optim.SGD(model.parameters(), lr=lrates[s])
  epochs=100
  loss_n=0.0
  loss_size=np.zeros(epochs)
  loss_val=np.zeros(epochs) 
  for e in range(epochs):
    running_loss=0.0
    for x,y in trainloader:
      optimizer.zero_grad()
      output=model(x)
      loss=criterion(output,y)
      loss.backward()
      optimizer.step()
      running_loss+=loss.item()
    loss_n+=running_loss/len(trainloader)
    loss_size[e]=loss_n/(e+1)
    valloss=0.0
    with torch.no_grad():
      for x,y in valloader:
        output=model(x)
        loss=criterion(output,y)
        valloss+=loss
    loss_val[e]=valloss/len(valloader)
  plt.plot(np.arange(1,101),loss_size,'b-',label='train cross entropy')
  plt.plot(np.arange(1,101),loss_val,'r-',label='val cross entropy')
  plt.legend()
