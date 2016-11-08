from __future__ import print_function

from pyspark import SparkContext, SparkConf
# $example on$
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint


def parseLine(line):
    parts = line.split(',')
    label = float(parts[len(parts)-1])
    
    features = Vectors.dense([float(parts[x]) for x in range(0,len(parts)-1)])
    return LabeledPoint(label, features)
# $example off$

if __name__ == "__main__":

    # Changed by me

    #sc = SparkConf().setMaster("local[1000]").setAppName("PythonNaiveBayes")
    #sc = SparkContext(sc1)
    sc = SparkContext(appName="PythonNaiveBayes")
    #sc = SparkContext(appName="PythonNaiveBayes",conf="local[1000]")

    # $example on$
    data = sc.textFile('a.csv').map(parseLine)

    # Split data aproximately into training (90%) and test (10%)
    training, test = data.randomSplit([0.9, 0.1], seed=0)

    # Train a naive Bayes model.
    model = NaiveBayes.train(training, 1.0)

    # Make prediction and test accuracy.
    predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
    accuracy = 1.0 * predictionAndLabel.filter(lambda x: x[0] == x[1]).count() / test.count()
    acc = accuracy * 100
    acc = str(acc)
    print("\n\n\nAccuracy is : " + acc + " % \n\n")
    # Save and load model
    #model.save(sc, "target/tmp/myNaiveBayesModel")
    #sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")
    # $example off$
