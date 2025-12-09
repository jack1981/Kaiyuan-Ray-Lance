DRIVER_MAX_RESULT_SIZE=8G
DRIVER_MEMORY=16G
EXECUTOR_MEMORY=16G
EXECUTOR_MEMORY_OVERHEAD=16G
EXECUTOR_CORES=4
EXECUTOR_INSTANCES=16

SCRIPT=$1
CONFIG_FILE=$2

export PYTHONPATH=./

${SPARK_HOME}/bin/spark-submit \
    --master local[*] \
    --conf spark.driver.maxResultSize=$DRIVER_MAX_RESULT_SIZE \
    --conf spark.driver.memory=$DRIVER_MEMORY \
    --conf spark.executor.memory=$EXECUTOR_MEMORY \
    --conf spark.executor.memoryOverhead=$EXECUTOR_MEMORY_OVERHEAD \
    --conf spark.executor.cores=$EXECUTOR_CORES \
    --conf spark.executor.instances=$EXECUTOR_INSTANCES \
    $1 \
    --config $CONFIG_FILE
