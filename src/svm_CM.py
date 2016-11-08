
from __future__ import print_function

from pyspark import SparkContext
# $example on$
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint


def parseLine(line):
    parts = line.split(',')
    label = float(parts[len(parts)-1])
    features = Vectors.dense([float(parts[x]) for x in range(0,len(parts)-1)])
    return LabeledPoint(label, features)
# $example off$

if __name__ == "__main__":

    sc = SparkContext(appName="PythonNaiveBayes")

    # $example on$
    data = sc.textFile('avg.csv').map(parseLine)

    # Split data aproximately into training (80%) and test (20%)
    training, test = data.randomSplit([0.8, 0.2],seed=0)
	
	#prepare model of svm on training data
    model = SVMWithSGD.train(training, iterations=100)

    # Make prediction and test accuracy.
    #to interate through rdd
    #predictionAndLabel.map(lambda x: print(str(x[0])+" "+str(x[1]))).collect()
    predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda x: x[0] == x[1]).count() / test.count()
    truepos=predictionAndLabel.filter(lambda x: (x[0]==1)and (x[1]==1)).count()
    trueneg=predictionAndLabel.filter(lambda x: (x[0]==0)and (x[1]==0)).count()
    falseneg=predictionAndLabel.filter(lambda x: (x[0]==0)and (x[1]==1)).count()
    falsepos=predictionAndLabel.filter(lambda x: (x[0]==1)and (x[1]==0)).count()
    print("\n\ntrue positive "+str(truepos))
    print("\n\ntrue negative "+str(trueneg))
    print("\n\nfalse negative "+str(falseneg))
    print("\n\nfalse pasitive "+str(falsepos))
    
    print("\n\n Accuracy is : " + str(accuracy * 100) + " % " + "\n\n\n" ) 
    