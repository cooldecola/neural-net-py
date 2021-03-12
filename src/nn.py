import numpy as np
import random
import math

LEARNING_RATE  = 0.5
ALPHA = 0.7

class Neuron:
    def __init__(self, n_index, layerNum, nextLayerNum):
        self.n_index = n_index # the index of the neuron
        self.layerNum=layerNum # what layer the neuron is in
        self.nextLayerNum=nextLayerNum # the number of neurons in the next layer
        self.weight_vec = [] # connection weights between self.Neuron and the neurons in the next layer
        self.delta_weight_vec = []
        self.outputVal = 1 # setting the bias output as 1
        self.gradient = 0 

    #Setting the weights of all neurons
    def setWeights(self):            
        self.weight_vec = [random.uniform(0,1) for i in range(self.nextLayerNum)]
        self.delta_weight_vec = [0 for i in range(self.nextLayerNum)]
        print (self.weight_vec)
        





class Layer(object):
    def __init__(self, layerNum, num_Neurons, nextLayerNum):
                        #what layer , how many neurons in this layer, how many neurons in the next layer
        self.layerNum = layerNum # index of what the layer number is
        self.num_Neurons = num_Neurons # number of neurons in the current layer
        self.nextLayerNum = nextLayerNum # number of neurons in the next layer
        self.neuron_vec = [] # vector that will contain the neurons in current layer

    # setting the weights of all neurons
    def setNeurons(self):
        print("This is Layer # " + str(self.layerNum + 1))
        print("Setting up neurons...")
        for i in range(self.num_Neurons + 1):  # plus one for the bias neuron
            if i < self.num_Neurons: 
                print ("Neuron # " + str(i+1), end="")    
            else:
                print ("Bias Neuron", end="") 
            n = Neuron(i, self.layerNum, self.nextLayerNum)
            n.setWeights()
            self.neuron_vec.append(n)
        print()


    def display(self): 
        for i in self.neuron_vec: 
            print(i.weight_vec)






class Network():
    def __init__(self, layer_vec):
        self.layer_vec = layer_vec.append(0)
        self.numOfLayers = len(layer_vec)-1
        self.layers = []


    def initialize_network(self):
        for i in range (self.numOfLayers):
            #print(i)
            l = Layer(i, layer_vec[i], layer_vec[i+1])
            l.setNeurons()
            #l.display()
            self.layers.append(l)

    def feedForward(self, input_neurons):
        #print(len(self.layers[0].neuron_vec)-1)
        if len(input_neurons) == len(self.layers[0].neuron_vec)-1: # checking if input neuron size = input neurons in net
            for layer in range (len(self.layers)):
                if layer == 0:
                    print("Calculating output values of input layer...")
                    for neuron in range(len(input_neurons)):
                        self.layers[layer].neuron_vec[neuron].outputVal = input_neurons[neuron]
                        print ("In Layer " + str(layer+1) + "; Neuron " + str(neuron+1) + " the output val is: " + 
                                str(self.layers[layer].neuron_vec[neuron].outputVal))
                        print()
                else:
                    if layer == 1: print("Calculating output values of hidden layers...")
                    sum = 0
                    for neuron, lyr in enumerate(self.layers[layer].neuron_vec[:-1]): #iterating over all the neurons in layer except bias
                        for prev_neuron, prv_lyr in enumerate(self.layers[layer-1].neuron_vec): # We include bias neuron from prev layer becuase it does impact the output of neurons in next layer
                            sum += prv_lyr.weight_vec[neuron] * prv_lyr.outputVal
                            print(str(prv_lyr.weight_vec[neuron]) + " * " + str(prv_lyr.outputVal))
                            print()

                        print(sum)
                        lyr.outputVal = activationFunction(sum)
                        print(lyr.outputVal)

    def backProp(self, target_values):

        #Calclating net Error
        error = 0
        outputlayerSize = len(target_values)
        gradient_output_vec = []
        for i in range(outputlayerSize):
            outputval = self.layers[-1].neuron_vec[i].outputVal
            #calculating gradient on output layer
            delta = target_values[i] - outputval
            tmp_grad = delta - activationFunction_derivative(outputval)
            self.layers[-1].neuron_vec[i].gradient = tmp_grad
            gradient_output_vec.append(tmp_grad)
            print(str(target_values[i]) + " - " + str(outputval))  
            print("Gradient for output = " + str(tmp_grad))
            error += delta * delta
        
        print ("error/" + str(len(self.layers[-1].neuron_vec) - 1))
        error = (error)/(len(self.layers[-1].neuron_vec) - 1)
        error = math.sqrt(error)
        print("RMS = " + str(error))
        print()

        #printing out gradient on output layer
        for i in range(outputlayerSize):
            print("GRAD = " +  str(self.layers[-1].neuron_vec[i].gradient))
        print()
        print("Calculating gradients for hidden layers...")

        #calculating gradient on hidden layers
        for lyr in reversed(self.layers[:-1]):
            sum_grad = 0
            layer_index = lyr.layerNum
            print("Calculating gradient on layer " + str(layer_index))
            for neuro in lyr.neuron_vec:
                print("Neuron # " + str(neuro.n_index + 1))
                for w in range(len(neuro.weight_vec)):
                    print(str(neuro.weight_vec[w]) + " * " + str(self.layers[layer_index + 1].neuron_vec[w].gradient))
                    sum_grad += neuro.weight_vec[w] * self.layers[layer_index + 1].neuron_vec[w].gradient
                
                neuro.gradient = sum_grad
                print("TOTAL GRAD = " + str(neuro.gradient))

    def updateWeights(self):
        #updating weights...
        print("updating weights...")

        for lyr in range( (len(self.layers)-2), -1, -1):
            print(lyr)
            print()

            for neuro in range(len(self.layers[lyr].neuron_vec)):

                for w in range(len(self.layers[lyr].neuron_vec[neuro].weight_vec)):
                    print (str(LEARNING_RATE) + " * " + str(self.layers[lyr].neuron_vec[neuro].outputVal) + " * " + str(self.layers[lyr].neuron_vec[neuro].gradient) + " + " + str(ALPHA) + " * " + str(w))
                    new_weight = LEARNING_RATE * self.layers[lyr].neuron_vec[neuro].outputVal * self.layers[lyr].neuron_vec[neuro].gradient + ALPHA * self.layers[lyr].neuron_vec[neuro].delta_weight_vec[w]                
                    print (" = " + str(new_weight))
                    print ("old weight of neuron " + str(self.layers[lyr].neuron_vec[neuro].n_index) + " = " + str(self.layers[lyr].neuron_vec[neuro].weight_vec[w]))
                    self.layers[lyr].neuron_vec[neuro].delta_weight_vec[w] = self.layers[lyr].neuron_vec[neuro].weight_vec[w]
                    self.layers[lyr].neuron_vec[neuro].weight_vec[w] += new_weight
        
"""         for lyr in reversed(self.layers[:-1]):

            for neuro in lyr.neuron_vec:
                new_weight_ls = []
                new_delta_w_ls = [] 

                for w, dw in zip(neuro.weight_vec, neuro.delta_weight_vec):
                    print (str(LEARNING_RATE) + " * " + str(neuro.outputVal) + " * " + str(neuro.gradient) + " + " + str(ALPHA) + " * " + str(dw))
                    new_weight = LEARNING_RATE * neuro.outputVal * neuro.gradient + ALPHA*dw
                    print (" = " + str(new_weight))
                    print("old weight of neuron " + str(neuro.n_index) + " = " + str(w))
                    dw = w
                    new_delta_w_ls.append(dw)
                    w += new_weight
                    new_weight_ls.append(w)
                    print(w)
                
            neuro.weight_vec = new_weight_ls
            neuro.delta_weight_vec = new_delta_w_ls """

     
        #print(self.layers[0].neuron_vec[0].weight_vec[0])
        #print(self.layers[0].neuron_vec[0].weight_vec[0])









def activationFunction(outputVal):
    val = math.tanh(outputVal)
    return val    

def activationFunction_derivative(val):
    return (1 - val*val)     




if __name__ == '__main__':
    
    
    layer_vec = [2,2,1]
    layer_num = len(layer_vec)

    net = Network(layer_vec)
    net.initialize_network()

    input_neurons = [1,0]

    print ("****************")
    print("Forward feeding ... ")
    print()
    net.feedForward(input_neurons)


    print("*****************")
    print("Back Propogation ... ")
    print()
    target_values = [0]
    net.backProp(target_values)

    net.updateWeights()

    input_neurons = [1,0]

    print ("****************")
    print("Forward feeding ... ")
    print()
    net.feedForward(input_neurons)


    print("*****************")
    print("Back Propogation ... ")
    print()
    target_values = [0]
    net.backProp(target_values)

    net.updateWeights()





    
"""     for i in range(layer_num):
        if i != layer_num-1: 
            l = Layer(i, layer_vec[i], layer_vec[i+1])
            l.setNeurons()
            l.setBias()
            l.display()
            layers.append(l)
        else: 
            l = Layer(i, layer_vec[i]) """
        
"""     print("*******************")
    print("Now calculating output values")

    

    for i in range(layer_num-1):
        print(len(layers))
        layers[i].feedForward(input_neurons) """
