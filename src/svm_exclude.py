from __future__ import print_function
#import test
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
    #accuracy1 = test.ret()
    # $example on$
    data = sc.textFile('exclude0.csv').map(parseLine)

    # Split data aproximately into training (90%) and test (10%)
    training, test = data.randomSplit([0.9, 0.1],seed=0)
	
	#prepare model of svm on training data
    model = SVMWithSGD.train(training, iterations=100)

    # Make prediction and test accuracy.
    predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda x: x[0] == x[1]).count() / test.count()

    #acc = accuracy1 * 100
    acc = accuracy * 100
    acc = str(acc)
    itr = str(100)
    print("\n\nNumber of iterations : " + itr)
    print("\n\n\nAccuracy is : " + acc + " % \n\n")
    # Save and load model
    #model.save(sc, "target/tmp/myNaiveBayesModel")
    #sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")
    # $example off$
    