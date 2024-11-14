#Cuda implementation of SVMs
import numpy as np
import torch

#https://www.pycodemates.com/2022/10/svm-kernels-polynomial-kernel.html

#SVM class
class SVC_CUDA:

    #Constructor -> take kernel and acceptable error E
    def __init__(self, err : int = 1.0 , kernel : str = "polynomial") -> None:
        
        #Set all torch tensors to be placed on the available GPU
        torch.cuda.set_device(0)
        torch.cuda.get_device_name()
        print(torch.cuda.get_device_name())

        #Set error and kernel
        self.err = err
        self.kernel = kernel

        #Set weight and biases
        self.weights = 0
        self.bias = 0

    #SVC uses hinge loss
    #Calculate hinge loss across entire dataset
    def hingeLoss(self, X_data, Y_data, weights, bias):
        
        #Optional regularization term
        #regterm = 

        #Loss term -> get accumulated loss
        loss = 0

        #Get predicted result via SVM
        for sample_idx in range(X_data.shape[0]):

            #Predicted result is dot product of weights and x, added to bias
            pred_res = torch.dot(X_data[sample_idx], weights) + bias
            
            #Hinge loss is max(0, 1 - pred_res * actual_res)
            #Multiply by acceptable error rate
            loss += self.err * (1 - max(0, 1 - (Y_data[sample_idx] * pred_res)))

        #Return accumulated loss
        return loss
    
    def convertLabels(self, Y_CUDA):
        zero_idxs = Y_CUDA == 0
        Y_CUDA[zero_idxs] = -1
        return Y_CUDA
    

    #Fit function
    #bs -> batch size, lr -> learning rate, n_epochs (number of epochs),
    #and init_epsilon (range around where tensor should be initialized)
    def fit(self, X_data, Y_data, bs, lr, n_epochs, init_epsilon):

        #Convert to torch tensor -> thanks to our device setting in the constructor,
        #this will move all respective data to the GPU 
        X_CUDA = torch.tensor(X_data, dtype = torch.float)
        Y_CUDA = self.convertLabels(torch.tensor(Y_data, dtype = torch.float))

        #Number of features and samples
        nFeat = X_CUDA.shape[1]
        nSamples = X_CUDA.shape[0]

        #Randomly shuffle samples before training
        #transformation to get random index
        get_rand = torch.randperm(nSamples)
        #Recreate x and y with shuffled rows
        X_CUDA = X_CUDA[get_rand]
        Y_CUDA = Y_CUDA[get_rand]

        #Initialze weights and biases to values close to zero
        #Initialize with normal distribution w/mean 0 and variance 1
        #Then, multiply by user-chosen constant
        weights = torch.randn(nFeat) * init_epsilon
        bias = torch.randn(1, 1) * init_epsilon

        print("WEIGHTS SHAPE", weights.shape)
        print("BIASES: SHAPE", bias.shape)

        #Store losses
        losses = []

        #Begin gradient descent
        for epoch_idx in range(n_epochs):
            #Execute each batch
            for batch_start_idx in range(0, nSamples, bs):

                #Accumulate gradients for each batch
                weights_grad = 0
                bias_grad = 0

                #Iterate through batch samples
                for sample_idx in range(batch_start_idx, batch_start_idx + bs):

                    #Could be a case where batch_start_idx is already at the end
                    #of the total number of samples
                    if (sample_idx >= nSamples): break

                    #Determine if the sample is classified correctly
                    #If prediction x actual label (labels are -1 or 1) > 1,
                    #then by definition, the predictions (when clipped in the range 
                    # -1 and 1) and labels were both identical
                    #If not, compute gradient
                    prod = torch.dot(X_CUDA[sample_idx], weights)
                    pred_label_product = Y_CUDA[sample_idx] * prod
                    
                    #So, only manipulate the gradients iff pred_label_product !> 1
                    if pred_label_product <= 1:

                        #Gradient of loss w.r.t. w is -c * Y[sample] * X[sample]
                        weights_grad -= self.err * Y_CUDA[sample_idx] * X_CUDA[sample_idx]
                        
                        #Gradient of loss w.r.t. b is -c * Y[sample]
                        bias_grad -= self.err * Y_CUDA[sample_idx]
            
                #Now, update weights and biases based on learning rate
                #Add regularization term -lr * w (acts as weight decay)
                weights = weights - lr * weights - lr * weights_grad
                bias = bias - lr * bias_grad

            #Compute loss and append
            loss = self.hingeLoss(X_CUDA, Y_CUDA, weights, bias)
            losses.append(loss)

        #Update weights and bias instance variables to post-training quantities
        self.weights = weights
        self.bias = bias

        #Return weights, bias, losses
        return self.weights, self.bias, losses

    #Prediction function
    def predict(self, X_data):

        #Move to cuda
        X_CUDA = torch.tensor(X_data, dtype = torch.float)
        #Predict
        pred = torch.matmul(X_CUDA, torch.reshape(self.weights, (2, 1)))
        print("SHAPE OF PREDICTIONS:", pred.shape)
        print("PREDICTIONS:", pred)
        return torch.squeeze(torch.sign(pred + self.bias)).cpu().detach().numpy()

    #Polynomial kernel
    def poly(self, X_data, Y_data):
        return
        #Transform x by adding new columns -> the square of each feature,
        #and the product of the two features


    

            

            
            


        
