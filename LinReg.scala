// Databricks notebook source
//PROJECT02-CSE-6332
//Name- Divya Darshi


import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, inv => breezeInv}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scala.math.sqrt

object LinReg extends Serializable {

  def computeSummand(w: Vector, labeledPoint: LabeledPoint): Vector = {
    val x = labeledPoint.features
    val y = labeledPoint.label
    val wBreeze = new BDV(w.toArray) // Convert to a Breeze dense vector
    val prediction = wBreeze.dot(new BDV(x.toArray)) // Dot product using Breeze
    val error = prediction - y
    Vectors.dense(x.toArray.map(_ * error)) // Convert back to Spark's Vector
  }

  def predict(w: Vector, labeledPoint: LabeledPoint): (Double, Double) = {
    val x = labeledPoint.features
    val y = labeledPoint.label
    val wBreeze = new BDV(w.toArray) // Convert to a Breeze dense vector
    val prediction = wBreeze.dot(new BDV(x.toArray)) // Dot product using Breeze
    (y, prediction)
  }

  def computeRMSE(predictions: RDD[(Double, Double)]): Double = {
    val n = predictions.count()
    val squaredErrors = predictions.map { case (y, prediction) => (y - prediction) * (y - prediction) }
    val sumSquaredErrors = squaredErrors.sum()
    val rmse = sqrt(sumSquaredErrors / n)
    rmse
  }

  def gradientDescent(data: RDD[LabeledPoint], iterations: Int, alpha: Double): (Vector, Array[Double]) = {
    val numFeatures = data.first().features.size
    var w = Vectors.dense(Array.fill(numFeatures)(0.0))
    val n = data.count()
    val errors = new Array[Double](iterations)

    for (i <- 0 until iterations) {
      val sumGrad = data.map(labeledPoint => computeSummand(w, labeledPoint)).reduce { (v1, v2) =>
        Vectors.dense((new BDV(v1.toArray) + new BDV(v2.toArray)).toArray)
      }

      val gradient = Vectors.dense(sumGrad.toArray.map(_ * (1.0 / n)))

      w = Vectors.dense(w.toArray.zip(gradient.toArray).map { case (w_i, grad_i) =>
        w_i - alpha / math.sqrt(i + 1) * grad_i
      })

      val predictions = data.map(labeledPoint => predict(w, labeledPoint))
      errors(i) = computeRMSE(predictions)
    }

    (w, errors)
  }

  def closedFormSolution(data: RDD[LabeledPoint]): Vector = {
    val numFeatures = data.first().features.size
    val X: BDM[Double] = new BDM(data.count().toInt, numFeatures, data.map(_.features.toArray).collect.flatten)
    val y: BDV[Double] = new BDV(data.map(_.label).collect)

    val XTX: BDM[Double] = X.t * X
    val XTy: BDV[Double] = X.t * y
    val XTXInverse: BDM[Double] = breezeInv(XTX)
    val w: BDV[Double] = XTXInverse * XTy

    Vectors.dense(w.data)
  }

  def main(args: Array[String]): Unit = {
    // Get or create the shared SparkContext
    val conf = new SparkConf().setAppName("LinearRegressionSpark")
    val sc = SparkContext.getOrCreate(conf)
    
    // Test the Part 1 - computeSummand function
    val wExample1 = Vectors.dense(Array(1.0, 2.0, 3.0))
    val labeledPoint1 = LabeledPoint(5.0, Vectors.dense(Array(2.0, 3.0, 4.0)))
    val summand1 = computeSummand(wExample1, labeledPoint1)
    println("Test the Part 1 - computeSummand function")
    println("Example 1 - Summand: " + summand1)

    val wExample2 = Vectors.dense(Array(0.5, 1.0, 1.5))
    val labeledPoint2 = LabeledPoint(3.0, Vectors.dense(Array(1.0, 2.0, 2.5)))
    val summand2 = computeSummand(wExample2, labeledPoint2)
    println("Example 2 - Summand: " + summand2)

    // Test the Part 2 - predict function
    val wPredict = Vectors.dense(Array(1.0, 2.0, 3.0))
    val labeledPointPredict = LabeledPoint(5.0, Vectors.dense(Array(2.0, 3.0, 4.0)))
    val (actualLabel, prediction) = predict(wPredict, labeledPointPredict)
    println("\nTest the Part 2 - predict function")
    println("Prediction: " + prediction + ", Actual Label: " + actualLabel)

    // Test the Part 3 - computeRMSE function
    val predictionsRDD = sc.parallelize(Seq((3.0, 2.5),(4.0, 3.8),(5.0, 5.2)))
    val rmse = computeRMSE(predictionsRDD)
    println("\nTest the Part 3 - computeRMSE function")
    println("RMSE: " + rmse)

    // Test the Part 4 - gradientDescent function
    val trainData = sc.parallelize(Seq(
      LabeledPoint(2.0, Vectors.dense(1.0, 2.0)),
      LabeledPoint(3.0, Vectors.dense(2.0, 3.0)),
      LabeledPoint(4.0, Vectors.dense(3.0, 4.0)),
    ))
    val numIterations = 5
    val alpha = 0.1
    val (weights, trainingErrors) = gradientDescent(trainData, numIterations, alpha)
    println("\nTest the Part 4 - gradientDescent function")
    println("Final Weights (Gradient Descent): " + weights)
    println("Training Errors:")
    trainingErrors.foreach(println)

    // Test the closed form solution on an example RDD
    val trainDataClosedForm = sc.parallelize(Seq(
      LabeledPoint(2.0, Vectors.dense(1.0, 2.0)),
      LabeledPoint(3.0, Vectors.dense(2.0, 3.0)),
      LabeledPoint(4.0, Vectors.dense(3.0, 4.0)),
    ))
    val weightsClosedForm = closedFormSolution(trainDataClosedForm)
    println("\nTest Part 5 - the closed form solution ")
    println("Closed Form Solution Weights: " + weightsClosedForm)

  }
}
LinReg.main(Array())


//REFERENCE:
//https://spark.apache.org/docs/latest/quick-start.html




