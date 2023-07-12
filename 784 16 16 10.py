#Eric LE
#Network Size: 784 x 16 x 16 x 10'
#Activation Function: Sigmoid function



import numpy as np
from keras.datasets import mnist
import copy as cp
from matplotlib import pyplot as plt

class NeuralNetwork():






    def __init__(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()
        #Initialise parameters for Weights and Biases
        WeightsLayer1 = np.subtract(np.random.rand(16,784),0.5,dtype=np.float16)
        WeightsLayer2 =np.subtract(np.random.rand(16,16),0.5,dtype=np.float16)
        WeightsOutputLayer = np.subtract(np.random.rand(10,16),0.5,dtype=np.float16)
        self.Weights = [WeightsLayer1,WeightsLayer2,WeightsOutputLayer]

        
        BiasLayer1 = np.random.randn(16)
        BiasLayer2 = np.random.randn(16)
        BiasOutputLayer = np.random.randn(10)
        self.Biases = [BiasLayer1,BiasLayer2,BiasOutputLayer]

        self.CorrectCounter = 0

    def Load_Data(self,i):
        #Input data from MNIST
        
        self.data1_data = self.train_images[i]
        self.data1_label = self.train_labels[i]

        self.correct = np.zeros(10)
        self.correct[self.data1_label] = 1
        



    def FeedForward(self,CurrentLayer):
        #Feed activations onto all layers, returning the output layer
        CurrentLayer = np.divide(np.reshape(CurrentLayer,(1,784))[0],256)
        self.AllLayers = [CurrentLayer]#4 items 
        self.UnactivatedLayers = [CurrentLayer]#4 items
        
        
        for WeightsOfLayer,BiasofLayer in zip(self.Weights,self.Biases):
            self.UnactivatedLayers.append(np.dot(WeightsOfLayer,CurrentLayer) + BiasofLayer)
            CurrentLayer = self.sigmoid(np.dot(WeightsOfLayer,CurrentLayer) + BiasofLayer)
            self.AllLayers.append(cp.deepcopy(CurrentLayer))
        return self.AllLayers[3]
        
    
    def CostFunction(self,a):
        self.cost = np.subtract(a,self.correct)
        self.costSum = np.sum(np.power(self.cost,2))
        if np.argmax(a) == self.data1_label:
            self.CorrectCounter += 1
    

    def sigmoid(self,a):
        return 1.0/(1.0+np.exp(-a))


    def sigmoid_deriv(self,a):
        return self.sigmoid(a)*(1-self.sigmoid(a))



    def BackPropagation(self):
        #Theta: Partial Derivative of Cost with respect to Change in Unactivated Layer
        #Start from outer layer-1
        Theta = np.multiply((self.cost),self.sigmoid_deriv(self.UnactivatedLayers[3]))  
        self.WeightChangeMatrix= [np.multiply(np.transpose([Theta]),[self.AllLayers[2]])]
        self.BiasChangeMatrix = [Theta]
        #Start Backpropagation for each hidden layer
        for i in range(len(self.Weights)-2,-1,-1):
            Theta = np.multiply(np.dot(np.transpose(self.Weights[i+1]),Theta),self.sigmoid_deriv(self.UnactivatedLayers[i+1]))
            self.BiasChangeMatrix.append(Theta)
            self.WeightChangeMatrix.append(np.multiply(np.transpose([Theta]),[self.AllLayers[i]]))
        self.WeightChangeMatrix = self.WeightChangeMatrix[::-1]
        self.BiasChangeMatrix = self.BiasChangeMatrix[::-1]


    def GradDescent(self,iterations,LearningRate):
        for iter in range(iterations):
            self.Load_Data(iter)
            self.FeedForward(self.data1_data)
            self.CostFunction(self.AllLayers[3])
            self.BackPropagation()

            self.Weights=np.subtract(self.Weights,np.multiply(LearningRate,self.WeightChangeMatrix))
            self.Biases=np.subtract(self.Biases,np.multiply(LearningRate,self.BiasChangeMatrix))


    def epochs(self,num,iterations,rate):
        for i in range(num):
            self.GradDescent(iterations,rate)
            print(f'Epoch: {i} Accuracy: {self.CorrectCounter/iterations+1}')


    def showdata(self):
        a=input()
        i=0
        while a!= "end":
            first_image = self.test_images[i]
            print(f'Prediction: {np.argmax(self.FeedForward(first_image))}\nCorrect:{self.test_labels[i]}')
            first_image = np.array(first_image, dtype='float')
            pixels = first_image.reshape((28, 28))
            plt.imshow(pixels, cmap='gray')
            plt.show()
            i+=1
            a=input()

            

net = NeuralNetwork()
net.epochs(5,59999,.25)
net.showdata()
    

