from pyspark.sql import SparkSession
import pyspark
from pyspark.sql.functions import hour, dayofweek
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StringType, DoubleType
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from math import sin, cos, atan, atan2, radians, sqrt, degrees
print("Libraries imported")

spark = SparkSession.builder.appName('Sendy_logistics').getOrCreate()
print("Spark session created")
sc = spark.sparkContext
sc.setLogLevel("ERROR")
print("====loading data=====")
data = spark.read.csv('s3://sendylogistics/Train(1).csv', inferSchema=True,header=True)
rider = spark.read.csv('s3://sendylogistics/Riders.csv', inferSchema=True,header=True)
print("=====Data loaded=====")

#Merge rider data to both train and test
data_merge = data.join(rider, on=['Rider Id'],how ='inner')
print("Megered data and rider together")

cols_to_drop = ['Vehicle Type','Order No','Arrival at Destination - Day of Month',
       'Arrival at Destination - Weekday (Mo = 1)',
        'Arrival at Destination - Time','Precipitation in millimeters',
                'Temperature','Rider Id', "User Id", ]

time_cols = ['Placement - Time',
             'Confirmation - Time',
             'Arrival at Pickup - Time',
             'Pickup - Time']

data_merge = data_merge.drop(*cols_to_drop)

#Extraxt hour and day of the week data from date columns

for col in time_cols:
    data_merge = data_merge.withColumn(col+"_Hour", hour(col))
print("Tine columns extracted")

#Calculate the harvsine distance
def harvsine_array(lat1, lng1 , lat2, lng2):
    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1 , lat2, lng2))
    AVG_EARTH_RADIUS = 6371 #in kilometers
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * atan(sqrt(d))
    return h


# Manhattan distance is the sum of the horizontal and vertical distance between points on a grid
def manhattan_dist(lat1 ,lng1, lat2, lng2):
    a = harvsine_array(lat1, lng1 , lat1 , lng2)
    b = harvsine_array(lat1,lng1, lat2, lng1)
    
    return a + b

#Direction from the given coordinates
def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))
    y = sin(lng_delta_rad) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lng_delta_rad)
    return degrees(atan2(y, x))

#Register the created functions
HarvsineUDF = udf(lambda a,b,c,d: harvsine_array(a,b,c,d),StringType())
ManhattanUDF = udf(lambda a,b,c,d: manhattan_dist(a,b,c,d),StringType())
BearingUDF = udf(lambda a,b,c,d: bearing_array(a,b,c,d),StringType())

print("Applying feature enginerring to dataframe")
# Call the harvsine function
data_merge = data_merge.withColumn("Harvsine Distance", HarvsineUDF("Pickup Lat",
                                                                    "Pickup Long", 
                                                                    "Destination Lat", 
                                                                    "Destination Long").cast(DoubleType()))

# Call the manhattan dist function
data_merge = data_merge.withColumn("Manhattan Distance", ManhattanUDF("Pickup Lat",
                                                                      "Pickup Long",
                                                                      "Destination Lat",
                                                                      "Destination Long").cast(DoubleType()))

# Call the bearing fnuction
data_merge = data_merge.withColumn("Direction Distance", BearingUDF("Pickup Lat",
                                                                    "Pickup Long",
                                                                    "Destination Lat",
                                                                    "Destination Long").cast(DoubleType()))

cols = data_merge.columns
data_merge = data_merge.drop(*time_cols)

#String indexer to convert cat column to double
cols_index = ["Personal or Business"]

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(data_merge) for column in cols_index]
pipeline_idx = Pipeline(stages = indexers)
final_df = pipeline_idx.fit(data_merge).transform(data_merge)
final_df = final_df.drop(*cols_index)

print("========String indexer applied========")

# train_data,test_data = final_df.randomSplit([0.7,0.3])
# print("Data frame split into train and test")
cols_to_vector = [cols for cols in final_df.columns if cols != 'Time from Pickup to Arrival']

assembler = VectorAssembler(
            inputCols = cols_to_vector,
            outputCol = "features"
    )
final_df = assembler.transform(final_df)
print("==================Vector assembler Done=======================")

train_data,test_data = final_df.randomSplit([0.7,0.3])
print("Data frame split into train and test")

gbt = GBTRegressor(featuresCol="features", 
                   labelCol="Time from Pickup to Arrival")
# final_df = assembler.transform(final_df)

paramGrid = ParamGridBuilder()\
            .addGrid(gbt.maxDepth, [2, 5])\
            .addGrid(gbt.maxIter, [10, 50])\
            .build()

#Next define evaluation metric.
evaluator = RegressionEvaluator(metricName="rmse",
                                labelCol=gbt.getLabelCol(),
                                predictionCol=gbt.getPredictionCol())

cv = CrossValidator(estimator=gbt, evaluator=evaluator,
                   estimatorParamMaps=paramGrid)

pipeline = Pipeline(stages=[cv])
print("===========Fitting data pipeline===========")
pipeline_model = pipeline.fit(train_data)
print("========Model trained=========")

predictions = pipeline_model.transform(test_data)
rmse = evaluator.evaluate(predictions)
print("RMSE : {}".format(rmse))
# print(f"RMSE :{rmse}")
pipeline_model.save("s3://sendylogistics/Saved_pipeline/v1")
print("pipeline saved")
