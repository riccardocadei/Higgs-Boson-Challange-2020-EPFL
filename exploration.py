# Useful starting lines
import numpy as np
from helpers import *
from methods import *
from process_data import *
from crossValidation import *
from exploration import *
from matplotlib import pyplot as plt    

seed=20

def featuresPlot(tX,featuresNames):
# Draw a figure for each feature, on which for each sample we plot its corresponding value
    for i in range(len(featuresNames)):
        idRed = np.arange(len(tX))[tX[:,i] == -999]
        idBlue = np.delete(np.arange(len(tX)),idRed)
        print(idRed)
        fig = plt.figure()
        plt.plot(idBlue,tX[idBlue,i],'b')
        plt.plot(idRed,np.ones(len(idRed))*np.min(tX[idBlue,i]),'r')
        plt.title(featuresNames[i], fontsize=12)
        plt.xlabel('rows',fontsize=12)
        plt.show()
    
    return 0

def distributionsPlot(y,tX,featuresNames):
    savetX = tX
    savey = y


    for i in range(len(featuresNames)):

        y =  savey[(savetX[:,i] != - 999.0)]
        tX = np.expand_dims(savetX[(savetX[:,i] != - 999.0),i],axis=1)

        idPositive = [y==1][0]
        idNegative = [y==-1][0]
        print(y.shape,tX.shape,idNegative.shape)
        upperNegQuantile = np.quantile(tX[idNegative,i],0.95,axis=0)
        lowerNegQuantile = np.quantile(tX[idNegative,i],0.05,axis=0)
        upperPosQuantile = np.quantile(tX[idPositive,i],0.95,axis=0)
        lowerPosQuantile = np.quantile(tX[idPositive,i],0.05,axis=0)

        upperQuantile = max(upperNegQuantile,upperPosQuantile)
        lowerQuantile = max(lowerNegQuantile,lowerPosQuantile)
        idNegative = [idNegative & (tX[:,i] < upperQuantile) & (tX[:,i]>lowerQuantile)][0]
        idPositive = [idPositive & (tX[:,i] < upperQuantile) & (tX[:,i]>lowerQuantile)][0]


        plt.hist(tX[idNegative,i]/len(tX) ,1000, histtype ='step',color='r',label='y == -1')

        plt.hist(tX[idPositive,i]/len(tX) ,1000, histtype ='step',color='b',label='y == 1')

        plt.legend(loc = "upper right")
        plt.title(featuresNames[i], fontsize=12)
        plt.show()
    return 0