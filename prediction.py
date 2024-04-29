import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def clean_data(data_frame):
    """Cleans data by casting columns to double and stripping extra quotes."""
    return data_frame.select(*(col(column).cast("double").alias(column.strip("\"")) for column in data_frame.columns))

if __name__ == "__main__":
    print("Starting Spark Application")

    spark_session = SparkSession.builder.appName("Wine-Quality-Prediction-SPARK-ML").getOrCreate()
    spark_context = spark_session.sparkContext
    spark_context.setLogLevel('ERROR')

    # Load the validation dataset from a local path
    local_path = "ValidationDataset.csv"  # Specify your local path here
    raw_data_frame = (spark_session.read
                      .format("csv")
                      .option('header', 'true')
                      .option("sep", ";")
                      .option("inferschema", 'true')
                      .load(local_path))

    # Clean and prepare the data
    clean_data_frame = clean_data(raw_data_frame)

    # Load the trained model from a local path
    trainedModelPath = "trainedmodel"  # Specify your local model path here
    predictionModel = PipelineModel.load(trainedModelPath)

    # Make predictions
    predictions = predictionModel.transform(clean_data_frame)

    # Select the necessary columns and compute evaluation metrics
    predictionResults = predictions.select(['prediction', 'label'])
    accuracyEvaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
    accuracy = accuracyEvaluator.evaluate(predictions)
    print(f'Test Accuracy of wine prediction model = {accuracy}')

    # F1 score computation using RDD API
    prediction_metrics = MulticlassMetrics(predictionResults.rdd.map(tuple))
    weighted_f1_score = prediction_metrics.weightedFMeasure()
    print(f'Weighted F1 Score of wine prediction model = {weighted_f1_score}')

    # Save the trained model back to S3
    trained_model_output_path = "/opt/trainedmodel"  # Specify your S3 path here
    predictionModel.write().overwrite().save(trained_model_output_path)

    print("Exiting Spark Application")
    spark_session.stop()
