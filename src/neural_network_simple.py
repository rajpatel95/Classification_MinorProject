# $example on$
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# $example off$
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("neural").config('spark.sql.warehouse.dir', 'file:///C:/Users/hp/Downloads/spark-2.0.0-bin-hadoop2.7/spark-2.0.0-bin-hadoop2.7/bin/').getOrCreate()

# $example on$
# Load training data
data = spark.read.format("libsvm").load("diabetes.libsvm")
# Split the data into train and test
splits = data.randomSplit([0.8, 0.2], 123)
train = splits[0]
test = splits[1]
# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [8, 20,12, 7, 5,4, 2]
# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=50, layers=layers, blockSize=128, seed=1234)
# train the model
model = trainer.fit(train)
# compute accuracy on the test set
result = model.transform(test)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

print("\n\n\nAccuracy: " + str(evaluator.evaluate(predictionAndLabels) * 100) + " % " + "\n\n\n\n")
# $example off$
spark.stop()
