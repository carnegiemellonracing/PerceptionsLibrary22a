#Import libraries
import numpy as np
import torch

#SVM class
class SVC_CUDA:

    """
    SVC CUDA - an implementation of polynomial-kernel SVMs from scratch with all operations on the GPU via PyTorch.

    INSTANCE VARIABLES:
    err (float) - regularization parameter C for SVM. Objective function for SVM is 1/2(||w||) + C\sum^n_i \gamma_i. 

          i) ||W|| is the magnitude of the vector perpendicular to the decision boundary, and so, must be minimized.
          ii) \sum^n_i \gamma_i are slack terms where n is the number of training samples. This is accumulated margin error 
              from decision boundary to data point for each training sample. \gamma_i is evaluated indirectly by checking KKT conditions (see .fit for more).
              Again, we'd like to minimize this term.
          iii) C (err) is thus a regularization parameter that trades between minimizing margin (w), or classifying al points correctly.

    kernel (str) - type of kernel this SVM implements. This is currently set to polynomial.
    degree (int) - degree of the polynomial decision boundary the SVM will learn.
    coeff - coefficient of the polynomial decision boundary we learn. Another hyperparameter.

    METHODS:    
    __init__(err, kernel, degree, float): constructor
    poly(X1 : torch.Tensor, X2 : torch.Tensor) -> (X1 matmul X2.T + coeff)^degree (torch.Tensor): computes polynomial kernel. For SVMS, X1 = X2 means we're 
                                                                                                  computing the symmetric gram matrix (polynomial similarity between every)
    convertLabels(Y_CUDA : torch.tensor) -> torch.Tensor: takes torch.tensor (moved to CUDA) of all labels and replaces all 0s with -1s for training.
    fit(X_data : npdarray, Y_data : npdarray, bs (int), lr (float), n_epochs (int)) -> alpha (torch.Tensor), bias (1x1 torch.Tensor), losses (arr of floats) : trains SVM on gpu based on provided npdrray data.
    predict(X_data : npdarray) -> npdarray: takes set of X data and returns SVM-generated classification labels.
    """

    #Constructor -> take kernel and acceptable error E
    def __init__(self, err : float = 1, kernel : str = "polynomial", degree: int = 3, coeff : float = 2.5) -> None:
        
        #Set all torch tensors to be placed on the available GPU
        torch.cuda.set_device(0)
        torch.cuda.get_device_name()
        print(torch.cuda.get_device_name())

        #Set error and kernel + degreef
        self.err = err
        self.kernel = kernel
        self.degree = degree
        self.coeff = coeff

        #Set alpha, weighs and biases
        self.weights = 0
        self.bias = 0
        self.alpha = None


    #Polynomial kernel
    #Takes in pytorch tensors that have already been migrated to CUDA
    def poly(self, X1, X2):
        return ((torch.matmul(X1, X2.T) + self.coeff) ** self.degree)
    
    #Provided input data has 0 and 1 labels. Must change to -1 and 1 to 
    #allow us to quickly classify a prediction sample pair as correct/incorrect
    #as a byproduct of evaluating the KKT condition 
    def convertLabels(self, Y_CUDA):
        zero_idxs = Y_CUDA == 0
        Y_CUDA[zero_idxs] = -1
        return Y_CUDA

    #Fit function
    #bs -> batch size, lr -> learning rate, n_epochs (number of epochs),
    #and init_epsilon (range around where tensor should be initialized)
    def fit(self, X_data, Y_data, bs, alpha_lr, bias_lr, n_epochs):

        #Convert to torch tensor -> thanks to our device setting in the constructor,
        #this will move all respective data to the GPU 
        X_CUDA = torch.tensor(X_data, dtype = torch.float)

        #Normalize X_CUDA to have zero mean and unit variance
        #Save to use during prediction
        self.x_mean = torch.mean(X_CUDA, dim = 0)
        self.x_std = torch.std(X_CUDA, dim = 0)
        X_CUDA = (X_CUDA - self.x_mean) / self.x_std
        #Replace all 0 labels with -1
        Y_CUDA = self.convertLabels(torch.tensor(Y_data, dtype = torch.float))

        #Randomly shuffle samples before training
        #Recreate x and y with shuffled rows
        nSamples = X_CUDA.shape[0]
        get_rand = torch.randperm(nSamples)

        #Apply shuffling (preserves entries across X and Y)
        X_CUDA = X_CUDA[get_rand]
        Y_CUDA = Y_CUDA[get_rand]

        #Store training data for later use
        self.X_train = X_CUDA
        self.Y_train = Y_CUDA

        #Initialize alpha and bias
        self.alpha = torch.zeros(nSamples, dtype = torch.float)
        self.bias = torch.zeros(1, dtype = torch.float)

        #Get kernel matrix
        kernelMat = self.poly(X_CUDA, X_CUDA)

        #Store losses
        losses = []

        #Begin gradient descent
        for epoch_idx in range(n_epochs):

            #Execute each batch
            for batch_start_idx in range(0, nSamples, bs):

                #End of batch
                batch_end_idx = min(batch_start_idx + bs, nSamples)
                #Alpha for this batch
                batch_alpha = self.alpha[batch_start_idx:batch_end_idx]
                #Kernel for this batch
                batch_kernel = kernelMat[batch_start_idx:batch_end_idx]
                #Labels for this batch
                batch_labels = Y_CUDA[batch_start_idx : batch_end_idx]

                #Get batch predictions
                batch_preds = torch.sum(self.alpha * Y_CUDA * batch_kernel, dim = 1) + self.bias

                #Update based on misclassified (byproduct of KKT condition checking)
                incorrectSamples = batch_labels * batch_preds <= 1
                #Update alpha and biases
                batch_alpha[incorrectSamples] += alpha_lr * self.err * (1 - batch_labels[incorrectSamples] * batch_preds[incorrectSamples])
                self.bias -= bias_lr * torch.sum(batch_labels[incorrectSamples])

                # #End batch
                # batch_end_idx = min(batch_start_idx + bs, nSamples)

                # #Iterate through batch samples
                # for sample_idx in range(batch_start_idx, batch_start_idx + bs):

                #     #Could be a case where batch_start_idx is already at the end
                #     #of the total number of samples
                #     if (sample_idx >= nSamples): break

                #     #Get prediction of SVM via prediction function
                #     pred = torch.sum(self.alpha * Y_CUDA * kernelMat[sample_idx]) + self.bias

                #     #Check if Karush-Kuhn-Tucker conditions met (series of first derivative tests)
                #     #Determine if the sample is classified correctly
                #     #If prediction x actual label (labels are -1 or 1) > 1,
                #     #then by definition, the predictions (when clipped in the range 
                #     # -1 and 1) and labels were both identical

                #     if Y_CUDA[sample_idx] * pred <= 1:
                        
                #         self.alpha[sample_idx] += torch.squeeze(alpha_lr * self.err * (1 - Y_CUDA[sample_idx] * pred))
                #         self.bias -= Y_CUDA[sample_idx] * bias_lr * self.err

            #Compute Hinge loss for this epoch
            predictions = torch.sum(self.alpha * Y_CUDA * kernelMat, dim=1) + self.bias
            hinge_loss = torch.mean(torch.clamp(1 - Y_CUDA * predictions, min=0))
            losses.append(hinge_loss.item())
            
            # loss = 0.0
            # for i in range(nSamples - 1):
            #     pred = torch.sum(self.alpha * Y_CUDA * kernelMat[i]) + self.bias
            #     loss += max(0, 1 - Y_CUDA[i] * pred)
            
            # #Store loss
            # losses.append(loss.item())

        #Return alpha (weights), bias, losses
        return self.alpha, self.bias, losses

    #Prediction function
    def predict(self, X_data):

        #Move to CUDA
        X_CUDA = torch.tensor(X_data, dtype = torch.float)
        #Normalize X_CUDA
        X_CUDA = (X_CUDA - self.x_mean) / self.x_std

        #Get kernel matrix
        kernelMat = self.poly(X_CUDA, self.X_train)

        #Predict
        pred = torch.sum(self.alpha * self.Y_train * kernelMat, dim = 1) + self.bias

        #Get signs of the predictions -> + entires become 1, - entires become -1, 0 entires remain 0
        signedMatrix = torch.sign(pred)

        #In our problem, the output expects 1s and 0s. So, map all -1 entries -> 0.
        signedMatrix[signedMatrix == -1] = 0 

        #Move result to cpu, detach from gradient tree, convert to numpy
        return torch.squeeze(signedMatrix).cpu().detach().numpy()