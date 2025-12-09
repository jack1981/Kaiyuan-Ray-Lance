from typing import List, Tuple

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, LongType


def weakly_connected_component_spark(
    edges_df: DataFrame, num_partitions: int
) -> DataFrame:
    edges = edges_df.rdd
    # generate links: src -> [dst1, dst2, ...]
    links = (
        edges.flatMap(lambda e: [(e[0], e[1]), (e[1], e[0])])
        .distinct(num_partitions)
        .groupByKey(num_partitions)
        .cache()
    )
    # generate ranks: src -> min(dst1, dst2, ...)
    ranks = links.mapValues(lambda nb: min(nb)).cache()
    initial_updates = ranks.count()
    delta = ranks
    current_updates = initial_updates

    def expand_msgs(tup: Tuple[int, Tuple[int, List[int]]]):
        src = tup[0]
        rank = tup[1][0]
        neighbors = tup[1][1]
        out = [(v, rank) for v in neighbors if v > rank]
        out.append((src, rank))
        return out

    while True:
        new_delta = (
            delta.join(links, num_partitions)
            .flatMap(expand_msgs)
            .reduceByKey(lambda l, r: min(l, r), num_partitions)
            .join(ranks, num_partitions)
            .filter(lambda tup: tup[1][0] < tup[1][1])
            .mapValues(lambda tup: tup[0])
            .cache()
        )
        current_updates = new_delta.count()
        new_ranks = (
            new_delta.fullOuterJoin(ranks, num_partitions)
            .map(
                lambda tup: (
                    (tup[0], tup[1][0])
                    if tup[1][0] is not None
                    else (tup[0], tup[1][1])
                )
            )
            .cache()
        )
        new_ranks.count()
        delta.unpersist()
        delta = new_delta
        ranks.unpersist()
        ranks = new_ranks
        print(f"Updates: {current_updates}")
        if current_updates == 0:
            break

    wcc_schema = StructType(
        [
            StructField("vid", LongType(), False),
            StructField("component", LongType(), False),
        ]
    )
    return edges_df.sparkSession.createDataFrame(ranks, wcc_schema)
