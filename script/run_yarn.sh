DRIVER_MAX_RESULT_SIZE=32G
DRIVER_MEMORY=32G
EXECUTOR_MEMORY=32G
EXECUTOR_MEMORY_OVERHEAD=32G
EXECUTOR_CORES=4
EXECUTOR_INSTANCES=800

SCRIPT=$1
CONFIG_FILE=$2

zip -r datafiner.zip datafiner

${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --conf spark.driver.maxResultSize=$DRIVER_MAX_RESULT_SIZE \
    --conf spark.driver.memory=$DRIVER_MEMORY \
    --conf spark.executor.memory=$EXECUTOR_MEMORY \
    --conf spark.executor.memoryOverhead=$EXECUTOR_MEMORY_OVERHEAD \
    --conf spark.executor.cores=$EXECUTOR_CORES \
    --conf spark.executor.instances=$EXECUTOR_INSTANCES \
    --conf spark.sql.autoBroadcastJoinThreshold=-1 \
    --conf spark.sql.legacy.parquet.nanosToMicros=true \
    --conf spark.sql.parquet.enableTimestampTypes=false \
    --conf spark.executorEnv.PYSPARK_PYTHON=$(which python) \
    --conf spark.pyspark.python=$(which python) \
    --conf spark.pyspark.driver.python=$(which python) \
    --py-files datafiner.zip \
    $1 \
    --config $CONFIG_FILE
