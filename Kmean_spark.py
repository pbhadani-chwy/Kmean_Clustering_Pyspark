from __future__ import print_function

import sys

import numpy as np
from numpy import array
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans


#def parseVectorAndDropLastColumn(line):
#    return np.array([float(x) for x in line.split(',')])

#**
#******** Parsing the data on the basis of comma and dropping the last column as it is the class value***#
def ParseAndDrop(inp_line):
    newarr=[]
    for k in range(len(inp_line.split(','))-1):
        newarr.append(float(inp_line.split(',')[k]))
    return np.array(newarr)
#*******************************************************************************************************#

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Not a correct no. of arguments", file=sys.stderr)
        exit(-1)
    #*********Initiating the spark context for the application Kmean *****************#
    sc = SparkContext(appName="KMeansApp")
    #************partitioning the data into RDD with the help of function .textFile and parsing it***************#
    inplines = sc.textFile('input.csv')
    inpdata = inplines.map(ParseAndDrop)
    # ********** k defines the number of cluster ********************************************#
    k = int(2)
    model = KMeans.train(inpdata, k)

    #***************reading the data points for calculating the accuracy and predicting the trained model **********#
    with open('Input.csv') as file:
       rows=file.readlines();


    #*************Initialization of the list *******************************************#
    DProw=[]
    PredVal_arrays=[]
    ActVal_arrays=[]
    #***********************************************************************************#
    for row in rows:
       row=row.rstrip("\n")

       #***********************Initializing an output list *************************************#
       Out=[]

       #******************adding dimension for prediction in output_data ***********************#
       for x in range(len(row.split(','))-1):
          Out.append(float(row.split(',')[x]))

       #*************using model.predict to predict the class of the test data in the output file****************#
       pred_class= float(model.predict(array(Out)))

       #*************adding the predicted and actual class to the output file *******************************#
       Out.append(pred_class)
       Out.append(float(row.split(',')[-1]))

       #
       # putting the actual and the real class value into two list for the calculation of the accuracy*******#
       PredVal_arrays.append(pred_class)
       ActVal_arrays.append(float(row.split(',')[-1]))
       DProw.append(Out)
       #*****************************************************************************************************#


# ******************* Inserting data into the outfile *******************************#
    a = np.asarray(DProw)
    np.savetxt("outfile.csv", a, delimiter=",",fmt="%1.3f")
# ************************************************************************************#

#****************** printing the predicted and actual value ****************************#
    print(PredVal_arrays)
    print(ActVal_arrays)
#**************************************************************************************#
    PdVal=np.array(PredVal_arrays)
    ActVal=np.array(ActVal_arrays)
#********* Assigning class to the data set *********************************************#
    cls1=np.array(ActVal[PdVal==0.0])
    cls2=np.array(ActVal[PdVal==1.0])

    pcnt= float(sum(cls1)/len(cls1))
#*************calculation of the accuracy************************************************#
    if pcnt <0.5:
    	predict_1 = np.array(PdVal == ActVal)
        act_match =predict_1[predict_1]
        accuracy= float((len(act_match) * 100)/len(ActVal))
        print('accuracy: ',str(accuracy))
    else:
        predict_1 = np.array(PdVal != ActVal)
        act_match =predict_1[predict_1]
        accuracy= float((len(act_match) * 100)/len(ActVal))
 	print('accuracy: ',str(accuracy))
#****************************************************************************************#
sc.stop()


