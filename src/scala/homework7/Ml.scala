package src.scala.homework7

import org.apache.spark.sql.{Encoders, SaveMode}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import src.scala.homework7.{Iris, SparkSessionWrapper}

object Main extends App
  with SparkSessionWrapper {

  val path_to_iris_data = "src/resources/data/IRIS.csv"
  val schema_iris_data = Encoders.product[Iris].schema

  import spark.implicits._

  // 1.Построена модель.
  var iris_data_raw = spark.read
    .option("header", "true")
    .schema(schema_iris_data)
    .csv(path_to_iris_data)
    .as[Iris]

  val assembler = new VectorAssembler()
    .setInputCols(Array("sepal_length","sepal_width","petal_length","petal_width"))
    .setOutputCol("features_vector")

  val iris_data = assembler.transform(iris_data_raw)

  val labelIndexer = new StringIndexer()
    .setInputCol("species")
    .setOutputCol("indexedLabel")
    .fit(iris_data)

  val featureIndexer = new VectorIndexer()
    .setInputCol("features_vector")
    .setOutputCol("indexedFeatures")
    .fit(iris_data)

  val Array(trainingData, testData) = iris_data.randomSplit(Array(0.7, 0.3))

  val rf = new RandomForestClassifier()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("indexedFeatures")
    .setNumTrees(10)

  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labelsArray(0))


  val pipeline = new Pipeline()
    .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

  val model = pipeline.fit(trainingData)

  // 2.Произведена оценка качества модели.
  val predictions = model.transform(testData)

  val evaluator_accuracy = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  val evaluator_precision = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("precisionByLabel")

  val evaluator_recall = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("recallByLabel")


  val accuracy = evaluator_accuracy.evaluate(predictions)
  val precision = evaluator_precision.evaluate(predictions)
  val recall = evaluator_recall.evaluate(predictions)

  println(s"Test accuracy = $accuracy")
  println(s"Test precision = $precision")
  println(s"Test recall = $recall")

  // 3.Модель сохранена на диск.
  model.write.overwrite.save("src/resources/data/model")

  spark.stop()
}
