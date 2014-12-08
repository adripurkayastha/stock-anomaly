'''
Created on Nov 23, 2014

@author: kooshag
'''


import pandas as pd
import numpy as np
from mock import inplace



filePath = '/Users/kooshag/Google Drive/code/data/'
fileName = 'SnP_consumer_dis_daily.csv'


df = pd.read_csv(filePath+fileName, 
                 sep =',', header = 0, parse_dates=[0], dayfirst=True,
                 index_col = 0, skiprows = [0,1,2,4])


#print(df[df.columns[0:10]]) #take the first 10 rows

#these values should be moved to a config file 
#overlap: size of overlap
#win: size of the window
#start = start+overlap


#for i in itr:
 #   getChunk()
    

def getChunk(start, win, frq):
    rng = -1  
    if frq == 'D':
        rng = pd.date_range(start, periods=win, freq='D')
        print (rng)
    elif frq == 'W':
        rng = pd.date_range(start, periods=win, freq='W')
    else:
        print ('ERROR: the frequency is not D nor W')
    
    
    
getChunk('1974-05-14', 365, 'D')




#-----------------
# Test
#----------------
#def testChopNRoll
#write tests to make sure the chopping is correct
 



    
        
    
    
  
        
        
        
        
        
