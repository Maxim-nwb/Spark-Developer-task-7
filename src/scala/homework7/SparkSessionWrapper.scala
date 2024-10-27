package src.scala.homework7

import org.apache.spark.sql.SparkSession

trait SparkSessionWrapper {

  lazy val spark: SparkSession = SparkSession
    .builder()
    .appName("SparkAPI")
    .master("local[*]")
    .getOrCreate
}
