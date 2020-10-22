# Useful starting lines
import numpy as np
from helpers import *
from methods import *
from process_data import *
from crossvalidation import *
from matplotlib import pyplot as plt    

seed=10

def featuresPlot(tX,featuresNames):
# Draw a figure for each feature, on which for each sample we plot its corresponding value
    for i in range(len(featuresNames)):
        idRed = np.arange(len(tX))[tX[:,i] == -999]
        idBlue = np.delete(np.arange(len(tX)),idRed)
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

        #[a,b] = np.histogram(tX[idNegative,i],100)
        ##plt.plot(np.polyval(np.polyfit(b[:-1], a/len(tX[idNegative,i]),10), (np.linspace(np.min(b),np.max(b),100))),color='r',label='y == -1')
        #plt.plot(b[:-1], a,color='r',label='y == -1')
        #[a,b] = np.histogram(tX[idPositive,i],100)
        ##plt.plot(np.polyval(np.polyfit(b[:-1], a/len(tX[idPositive,i]),10), (np.linspace(np.min(b),np.max(b),100))),color='b',label='y == 1')
        #plt.plot(b[:-1], a,color='b',label='y == 1')        
        #plt.legend(loc = "upper right")
        #plt.title("{name}, feature: {id}/{tot}".format(name=featuresNames[i],id=i,tot=len(featuresNames)), fontsize=12)
        #plt.show()
            print(tX[:,i].shape)
            plt.hist(tX[idPositive,i] ,100, histtype ='step',color='b',label='y == 1',density=True)      
            plt.hist(tX[idNegative,i] ,100, histtype ='step',color='r',label='y == -1',density=True)  
            plt.legend(loc = "upper right")
            plt.title("{name}, feature: {id}/{tot}".format(name=featuresNames[i],id=i,tot=len(featuresNames)-1), fontsize=12)
            plt.show()
    return 0

def correlationMatrix(tX):
    return np.cov(tX.T)

def featuresVariance(tX,featuresNames):
    s = np.diag(correlationMatrix(tX))
    for i in range(len(s)):
        print("{num}/{tot} {name}: {var}".format(num=i,tot=len(s)-1,name=featuresNames[i],var=s[i]))
    return 0