import numpy as np
import random


def fMSE_loss(x,y,w):
    n = x.shape[0]
    y_predict = np.dot(x,w)
    fmse_loss = sum((y_predict-y)**2)/(2*n)   
    return fmse_loss

def mini_batch_SGD_Ridge(x,y,lr,lamda,epoch,batch_size):
    
    # set w to random values
    w=np.random.rand(x.shape[1],)
    
    # randomize the order of the training data(x,y) with same random seed
    seed = np.random.randint(100)
    random.Random(seed).shuffle(x)
    random.Random(seed).shuffle(y)
    
    # get the number of samples
    n_tr = x.shape[0]
    n_te = x.shape[0]
    
    for e in range(epoch):        
        for i in range(0,n_tr,batch_size):          
            # sample the training data based on the mini-batch size
            x_i = x[i:i+batch_size]
            y_i = y[i:i+batch_size]
                      
            # update weights base on L2 loss function
            
            #fmse_l2(w) = 1/2n * sum(y-y*)^2 + lamda/2* w.T@w
            # ∂fmse/∂w = 1/n *X@(X.T@w -y) + lamda*w       
            # w = w(1- a*(lamda/n)) - a* 1/n *X@(X.T@w -y)
            
            w = w*(1-lr*(lamda/n_tr)) - lr * 1/(n_tr)*np.dot(x.T,np.dot(x,w)-y)
        # compute MSE loss in the training data   
        fMSE_cost_tr = fMSE_loss(x,y,w)

        # report training loss every 10 epochs
        if e % 10 == 0:
            print ("epoch: " +str(e)+"  fMSE cost on the training data is: "+ str(fMSE_cost_tr))
    return w


def train():
    
    # load training and testing data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    # add bias term into each X
    X_tr_b = np.c_[np.ones((5000, 1)), X_tr]
    X_te_b = np.c_[np.ones((2500, 1)), X_te]

    # randomly split 20% data to create validation set from training set
    seed = np.random.randint(100)
    random.Random(seed).shuffle(X_tr_b)
    random.Random(seed).shuffle(ytr)

    remain=int(len(X_tr)*0.8)
    X_train, X_valid = X_tr_b[:remain,:], X_tr_b[remain:,:]
    y_train, y_valid = ytr[:remain], ytr[remain:]

    X_test, y_test = X_te_b,yte


    # create alternative sets of hyperparameters
    lr_set=[0.0001,0.0002,0.0005,0.001]
    lamda_set=[0.001,0.01,0.02,0.05]
    epoch_set=[20,30,40,50]
    batch_size_set=[20,30,50,100]

    # tuning hyperparameters
    Hyperparameter=[]
    for lr in lr_set:
        for lamda in lamda_set:
            for epoch in epoch_set:
                for batch_size in batch_size_set:
                    
                    print ("-----------------------------------------------")
                    print ('Hyperparameter: lr: {} lamda: {} epoch: {} batch size: {}'.format(str(lr),str(lamda),str(epoch),str(batch_size)))
                    w=mini_batch_SGD_Ridge(X_train,y_train,lr,lamda,epoch,batch_size)
                    
                    # compute unregularized MSE loss in the validation set
                    Validation_error = fMSE_loss(X_valid,y_valid,w)
                    print ("Validation cost of the combination of hyperparameters is: {}".format(str(Validation_error)))
                    
                    # save parameter combination,loss and w
                    hyperparameter=[[lr,lamda,epoch,batch_size],[Validation_error],[w]]                
                    Hyperparameter.append(hyperparameter)
    print ("==============================================")
    print ("Model training completed")
    print ("==============================================")

    # find the combination of hyperparameters has minimal validation error
    minimum = float('inf')
    best =0
    for i in Hyperparameter:
        if i[1][0]<minimum:
            minimum=i[1][0]
            best=i
    lr_best=best[0][0]
    lamda_best=best[0][1]
    epoch_best=best[0][2]
    batch_size_best=best[0][3]
    w_best=best[2][0]

    print ('The best combination of hyperparameters is: lr: {} lamda: {} epoch: {} batch size: {}'.format(str(lr_best),str(lamda_best),str(epoch_best),str(batch_size_best)))

    # compute the MSE loss with best parameter in the testing data
    test_error = fMSE_loss(X_test,y_test,w_best)

    # report final result
    print ("The fMSE cost on the testing data is: {}".format(str(test_error)))
