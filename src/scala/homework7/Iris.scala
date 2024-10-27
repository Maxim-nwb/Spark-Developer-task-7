package src.scala.homework7

import org.apache.spark.sql.types.Decimal


case class Iris(
                 sepal_length: Decimal,
                 sepal_width: Decimal,
                 petal_length: Decimal,
                 petal_width: Decimal,
                 species: String
               )
