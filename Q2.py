import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import torchvision
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import pickle
class MyNeuralNetwork():
    """
    My implementation of a Neural Network Classifier.
    """

    acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
    weight_inits = ['zero', 'random', 'normal']

    def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs, X_test=None, y_test=None):
        """
        Initializing a new MyNeuralNetwork object

        Parameters
        ----------
        n_layers : int value specifying the number of layers

        layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

        activation : string specifying the activation function to be used
                     possible inputs: relu, sigmoid, linear, tanh

        learning_rate : float value specifying the learning rate to be used

        weight_init : string specifying the weight initialization function to be used
                      possible inputs: zero, random, normal

        batch_size : int value specifying the batch size to be used

        num_epochs : int value specifying the number of epochs to be used
        """

        self.n_layers=n_layers
        self.layer_sizes=layer_sizes
        self.activation=activation
        self.learning_rate=learning_rate
        self.weight_init=weight_init
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        self.init_weights()
        self.values={}
        self.trainloss=np.zeros(self.num_epochs)
        self.testloss=np.zeros(self.num_epochs)
        self.trainacc=np.zeros(self.num_epochs)
        self.testacc=np.zeros(self.num_epochs)
        self.X_test=X_test
        self.y_test=np.squeeze(np.eye(10)[y_test.astype(np.int).reshape(-1)])


        if activation not in self.acti_fns:
            raise Exception('Incorrect Activation Function')

        if weight_init not in self.weight_inits:
            raise Exception('Incorrect Weight Initialization Function')

    def init_weights(self):
      '''initializes the weights'''
      self.weights=[]
      if self.weight_init=='normal':
        for i in range(self.n_layers-1):
          w=self.normal_init((self.layer_sizes[i],self.layer_sizes[i+1]))
          # self.weights.append(w/(w.max()*100))
          self.weights.append(w*0.01)
      elif self.weight_init=='random':
        for i in range(self.n_layers-1):
          w=self.random_init((self.layer_sizes[i],self.layer_sizes[i+1]))
          # self.weights.append(w/(w.max()*100))
          self.weights.append(w*0.01)
      elif self.weight_init=='zero':
        for i in range(self.n_layers-1):
          self.weights.append(zero_init((self.layer_sizes[i],self.layer_sizes[i+1])))
      self.weights=np.asarray(self.weights)
      self.bias=np.zeros(self.n_layers-1)



    def init_layers(self,size):
      '''initialize the empty layers of corresponding sizes'''
      self.values=[np.empty((size,layer)) for layer in self.layer_sizes]
      self.acts=[np.empty((size,layer)) for layer in self.layer_sizes]

    # def forward(self,X):
    #   h1=copy.deepcopy(X)
    #   self.values[0]=h1
    #   for i,weight in enumerate(self.weight):
    # def forward(self,X):
    #   h1=copy.deepcopy(X)
    #   self.values[0]=h1
    #   for i,weight in enumerate(self.weights):
    #     if self.activation=='sigmoid':
    #       h1=self.sigmoid(h1@weight)
    #     if self.activation=='linear':
    #       h1=self.linear(h1@weight)
    #     if self.activation=='relu':
    #       h1=self.relu(h1@weight)
    #     if self.activation=='tanh':
    #       h1=self.tanh(h1@weight)
    #     self.values[i+1]=h1
    #   self.output=self.softmax(self.values[-1])

    def forward(self, X):
      '''performs forward propogation step'''
      h1=copy.deepcopy(X)
      self.values[0]=h1
      self.acts[0]=h1
      for i in range(self.n_layers-2):
        weight=self.weights[i]
        l1=h1.dot(weight)+self.bias[i]
        if self.activation=='sigmoid':
          h1=self.sigmoid(l1)
        elif self.activation=='tanh':
          h1=self.tanh(l1)
        elif self.activation=='relu':
          h1=self.relu(l1)
        elif self.activation=='linear':
          h1=self.linear(l1)
        elif self.activation=='softmax':
          h1=self.softmax(l1)
        self.values[i+1]=h1
        self.acts[i+1]=l1
      l1=self.values[-2].dot(self.weights[-1])+self.bias[-1]
      self.acts[-1]=l1
      self.values[-1]=self.softmax(l1)
      # self.output=self.softmax(self.values[-1])
      self.output=self.values[-1]






    def back_prop(self, y):

      '''performs backpropogation step'''
      delta=(self.output-y)
      if self.activation=='sigmoid':
        # delta=delta*self.sigmoid_grad(self.values[-1])
        for i in range(1,self.n_layers):
          self.weights[-i]-=self.learning_rate*(np.dot(self.values[-i-1].T,delta))/self.batch_size
          self.bias[-i]-=self.learning_rate*np.mean(delta)/self.batch_size
          delta=self.sigmoid_grad(self.acts[-i-1])*(np.dot(delta,self.weights[-i].T)+self.bias[-i])

      elif self.activation=='relu':
        # delta=delta*self.relu_grad(self.values[-1])
        for i in range(1,self.n_layers):
          self.weights[-i]-=self.learning_rate*(np.dot(self.values[-i-1].T,delta))/self.batch_size
          self.bias[-i]-=self.learning_rate*np.mean(delta)/self.batch_size
          delta=self.relu_grad(self.acts[-i-1])*(np.dot(delta,self.weights[-i].T)+self.bias[-i])
          
      elif self.activation=='linear':
        # delta=delta*self.linear_grad(self.values[-1])
        for i in range(1,self.n_layers):
          self.weights[-i]-=self.learning_rate*(np.dot(self.values[-i-1].T,delta))/self.batch_size
          self.bias[-i]-=self.learning_rate*np.mean(delta)/self.batch_size
          delta=self.linear_grad(self.acts[-i-1])*(np.dot(delta,self.weights[-i].T)+self.bias[-i])
          
      elif self.activation=='tanh':
        # delta=delta*self.tanh_grad(self.values[-1])
        for i in range(1,self.n_layers):
          self.weights[-i]-=self.learning_rate*(np.dot(self.values[-i-1].T,delta))/self.batch_size
          self.bias[-i]-=self.learning_rate*np.mean(delta)/self.batch_size
          delta=self.tanh_grad(self.acts[-i-1])*(np.dot(delta,self.weights[-i].T)+self.bias[-i])
         
      elif self.activation=='softmax':
        # delta=delta*self.softmax_grad(self.values[-1])
        for i in range(1,self.n_layers):
          self.weights[-i]-=self.learning_rate*(np.dot(self.values[-i-1].T,delta))/self.batch_size
          self.bias[-i]-=self.learning_rate*np.mean(delta)/self.batch_size
          delta=self.softmax_grad(self.acts[-i-1])*(np.dot(delta,self.weights[-i].T)+self.bias[-i])
      
        




        



        


    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        x_calc=copy.deepcopy(X)
        x_calc=np.where(x_calc<0,0,x_calc)
            


        return x_calc

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        x_calc=copy.deepcopy(X)
        x_calc=np.where(x_calc<0,0,1)
        return x_calc

    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        # newX=X-np.max(X)
        newX=X
        return 1./(1.+np.exp(-newX))
    
    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        h=self.sigmoid(X)
        return h*(1.-h)

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        # self.linear_g=random.random()/100
        return X

    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.ones(X.shape)

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        # newX=X-np.max(X)
        newX=X
        a=np.exp(newX)
        b=np.exp(-newX)
        x_calc=a+b
        # x_calc=np.where(x_calc==0,0.1,x_calc)

        return (a-b)/x_calc

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1.-(self.tanh(X))**2

    def softmax(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        # newX=X-X.max()
        newX=X
        exponent=np.exp(newX)
        return exponent/exponent.sum(axis=1,keepdims=True)

    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        s=np.sum(np.exp(X),axis=0)
        a=s-np.exp(X)
        return a/(s**2)



    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.zeros(shape)

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.random.rand(shape[0],shape[1])

    def normal_init(self, shape):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.random.normal(size=shape)



    def loss(self,y_pred,y):
      '''returns the cross entropy loss'''
      epsilon=1e-12
      y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
      return ((-np.log(y_pred))*y).sum(axis=1).mean()
    



      
    def fit(self, X, y):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """
        n_samples=len(y)
        y=np.squeeze(np.eye(10)[y.astype(np.int).reshape(-1)])
        # print(y.shape)
        
        for e in range(self.num_epochs):
          print(e)
          self.init_layers(self.batch_size)
          # for i in range(self.n_layers-1):
          # #   # maxi=self.weights[i].max()
          # #   # if(max!=0):
          #   self.weights[i]=self.weights[i]/self.weights[i].max()
            
          # maxi=self.bias.max()
          # if maxi!=0:
          #   self.bias=self.bias/self.bias.max()
            # print(w,self.weights[i].max())
          # self.weights=self.weights/np.max(np.amax(self.weights))
          training_loss=0.0
          training_acc=0.0
          perm=np.random.permutation(n_samples)
          X_batches=np.array_split(X[perm],n_samples/self.batch_size)
          y_batches=np.array_split(y[perm],n_samples/self.batch_size)
          for x_train,y_train in zip(X_batches,y_batches):
            self.forward(x_train)
            # print(self.output)
            # training_loss+=self.loss(self.output,y_train)
            training_loss+=self.loss(self.values[-1],y_train)
            training_acc+=self.accuracy(self.output,y_train)

            # print()
            self.back_prop(y_train)
            # self.backpropogation(y_train)
          self.trainloss[e]=training_loss/len(y_batches)
          self.trainacc[e]=training_acc/len(y_batches)
          if self.X_test is not None and self.y_test is not None:
            self.init_layers(len(self.y_test))
            self.forward(self.X_test)
            # y_t=np.squeeze(np.eye(10)[self.y_test.astype(np.int).reshape(-1)])
            self.testloss[e]=(self.loss(self.output,self.y_test))
            self.testacc[e]=self.accuracy(self.output,self.y_test)



        # fit function has to return an instance of itself or else it won't work with test.py
        return self
    def accuracy(self,y_pred,y):
      '''calculates accuracy'''
        a=np.argmax(y_pred,axis=1)
        a=np.squeeze(np.eye(10)[a.astype(np.int).reshape(-1)])
        return np.all(a==y,axis=1).mean()


    def predict_proba(self, X):
        """
        Predicting probabilities using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the prediction probabilities.
        """

        # return the numpy array y which contains the predicted values
        self.init_layers(len(X))
        self.forward(X)
        return self.output

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """


        # return the numpy array y which contains the predicted values
        self.init_layers(len(X))
        self.forward(X)
        
        return np.argmax(self.output,axis=1)

    def score(self, X, y):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

        Returns
        -------
        acc : float value specifying the accuracy of the model on the provided testing set
        """

        # return the numpy array y which contains the predicted values
        self.init_layers(len(X))
        self.forward(X)
        return self.accuracy(self.output,y)
dftrain=pd.read_csv('/content/drive/MyDrive/ML DataSet/mnist_train.csv',header=None)
X_train=dftrain.iloc[:,1:].to_numpy()
y_train=dftrain.iloc[:,0].to_numpy()
X_train,ax,y_train,ay=train_test_split(X_train,y_train,test_size=0.5,stratify=y_train)
dftest=pd.read_csv('/content/drive/MyDrive/ML DataSet/mnist_test.csv',header=None)
X_test=dftest.iloc[:,1:].to_numpy()
y_test=dftest.iloc[:,0].to_numpy()
X_test,ax,y_test,ay=train_test_split(X_test,y_test,test_size=0.5,stratify=y_test)
X_train=X_train/X_train.max()
X_test=X_test/X_test.max()
y_test=np.squeeze(np.eye(10)[y_test.astype(np.int).reshape(-1)])
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
acti_fns = ['linear','tanh', 'sigmoid', 'relu']
# acti_fns = [ 'sigmoid']
acc_arr=np.zeros(len(acti_fns))
# saved_wts=[]
# saved_bias=[]
for i,fn in enumerate(acti_fns):
  print(fn)
  plt.figure()
  NN=MyNeuralNetwork(5,[784,256,128,64,10],fn,0.1,'normal',1500,100,X_test=X_test,y_test=y_test)
  NN.fit(X_train,y_train)
  print(np.argmax(NN.output, axis=1))
  accuracy=NN.score(NN.X_test,NN.y_test)
  print(accuracy)
  acc_arr[i]=accuracy
  filename=fn+'.sav'
  pickle.dump(NN, open(filename, 'wb'))
  # saved_wts.append(NN.weights)
  # saved_bias.append(NN.bias)
  plt.plot(np.arange(1,NN.num_epochs+1),NN.trainloss,label='loss-train'+fn)
  plt.plot(np.arange(1,NN.num_epochs+1),NN.testloss,label='loss-test'+fn)  
  plt.legend()
  plt.show()




max_fn=acti_fns[np.argmax(acc_arr)]
filename=max_fn+'.sav'
NN=pickle.load(open(filename, 'rb'))
input=X_train
for i in range(len(wts)-1):
  input=input@NN.weights[i]+NN.bias[i]

def TSNE_analysis(X,y):
  tsne=TSNE(n_components=2)
  df=pd.DataFrame()
  X_tsne=tsne.fit_transform(X)
  perm=np.random.permutation(X.shape[0])
  df['tsne_one']=X_tsne[perm,0]
  df['tsne_two']=X_tsne[perm,1]
  plt.figure(figsize=(16,10))
  sns.scatterplot(
      x="tsne_one", y="tsne_two",
      hue=y.ravel()[perm],
      palette=sns.color_palette("hls", 10),
      data=df,
      legend="full",
      alpha=1.0
  )
TSNE_analysis(input,y_train)
