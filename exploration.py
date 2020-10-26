import numpy as np
from helpers import *
from implementations import *
from process_data import *
from crossvalidation import *
import matplotlib.pyplot as plt    


def distributionsPlot(y,tX,featuresNames):
    savetX = tX
    savey = y
    alphaQuantile = 0

    for i in range(len(featuresNames)):

        y =  savey[(savetX[:,i] != - 999.0)]
        tX = savetX[(savetX[:,i] != - 999.0),:]
        
        if tX.shape[0]!=0:

            idPositive = [y==1][0]
            idNegative = [y==-1][0]
            upperNegQuantile = np.quantile(tX[idNegative,i],1-alphaQuantile,axis=0)
            lowerNegQuantile = np.quantile(tX[idNegative,i],alphaQuantile,axis=0)
            upperPosQuantile = np.quantile(tX[idPositive,i],1-alphaQuantile,axis=0)
            lowerPosQuantile = np.quantile(tX[idPositive,i],alphaQuantile,axis=0)

            upperQuantile = max(upperNegQuantile,upperPosQuantile)
            lowerQuantile = max(lowerNegQuantile,lowerPosQuantile)
            idNegative = [idNegative & (tX[:,i] <= upperQuantile) & (tX[:,i]>=lowerQuantile)][0]
            idPositive = [idPositive & (tX[:,i] <= upperQuantile) & (tX[:,i]>=lowerQuantile)][0]

            plt.hist(tX[idPositive,i] ,100, histtype ='step',color='b',label='y == 1',density=True)      
            plt.hist(tX[idNegative,i] ,100, histtype ='step',color='r',label='y == -1',density=True)  
            plt.legend(loc = "upper right")
            plt.title("{name}, feature: {id}/{tot}".format(name=featuresNames[i],id=i,tot=len(featuresNames)-1), fontsize=12)
            plt.show()

def featuresplot(tX,featuresNames):
# Draw a figure for each feature, on which for each sample we plot its corresponding value
    for i in range(len(featuresNames)):
        idRed = np.arange(len(tX))[tX[:,i] == -999]
        idBlue = np.delete(np.arange(len(tX)),idRed)
        fig = plt.figure()
        plt.plot(idBlue,tX[idBlue,i],'b')
        plt.title(featuresNames[i], fontsize=12)
        plt.xlabel('rows',fontsize=12)
        plt.show()

def class_in_training_set_plot(y,tX):
    msk_jets_train = {
        0: tX[:, 22] == 0,
        1: tX[:, 22] == 1,
        2: tX[:, 22] == 2, 
        3: tX[:, 22] == 3
        }

    ax = plt.subplot(111)
    colors = ['b','g','r','y']
    legend = ['calss: 0','class: 1','class: 2','class: 3']
    ind = np.array([-1,  1])
    w = 0.25
    for idx in range(len(msk_jets_train)):
        y_idx = y[msk_jets_train[idx]]
        count_prediction = {-1:  np.count_nonzero(y_idx == -1), 1:  np.count_nonzero(y_idx == 1)}
        ax.bar(ind+w*idx, count_prediction.values(), width=w, color=colors[idx],align='center')

    ax.set_ylabel('Numbers of training data')
    ax.set_xticks(ind+0.25)
    ax.set_xticklabels( ('prediction is -1', 'prediction is 1') )
    ax.legend(legend)
    ax.plot()