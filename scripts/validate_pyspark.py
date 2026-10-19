#!/usr/bin/env python3
"""
TrafficPulse - PySpark Environment Validation Script
Tests PySpark session creation, DataFrame transformations, and JVM bridge connectivity.
"""
import os
import sys
import shutil

def main():
    print("=" * 80)
    print(" TrafficPulse - PySpark Engine Validation")
    print("=" * 80)

    # 1. Check Java / JVM prerequisite
    java_path = shutil.which("java")
    java_home = os.environ.get("JAVA_HOME")
    print(f"[CHECK] Java executable in PATH: {java_path or 'NOT FOUND'}")
    print(f"[CHECK] JAVA_HOME environment:   {java_home or 'NOT CONFIGURED'}")

    if not java_path and not java_home:
        print("\n[BLOCKED] PySpark requires a Java Virtual Machine (JDK 8, 11, or 17).")
        print("          Neither 'java' was found in PATH nor is 'JAVA_HOME' defined.")
        print("          To resolve: Install OpenJDK (e.g. winget install Microsoft.OpenJDK.17) or run via container.")
        return 1

    # 2. Check PySpark import
    try:
        import pyspark
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import avg, count, round as spark_round
        print(f"[CHECK] PySpark package version:  {pyspark.__version__}")
    except ImportError as e:
        print(f"\n[BLOCKED] PySpark is not installed in the active Python environment: {e}")
        return 2

    # 3. Initialize minimal SparkSession
    print("\n[INIT] Initializing local SparkSession...")
    try:
        spark = SparkSession.builder \
            .appName("TrafficPulse_Validation") \
            .master("local[1]") \
            .config("spark.driver.memory", "512m") \
            .config("spark.sql.shuffle.partitions", "2") \
            .getOrCreate()
        print(f"[STATUS] SparkSession active! Version: {spark.version}")

        # 4. Create small test DataFrame
        sample_data = [
            (161, "Midtown Center", "Manhattan", 9.45, 1420),
            (237, "Upper East Side South", "Manhattan", 11.20, 1180),
            (132, "JFK Airport", "Queens", 24.80, 890),
            (138, "LaGuardia Airport", "Queens", 21.30, 940),
            (79, "East Village", "Manhattan", 10.15, 760)
        ]
        columns = ["location_id", "zone_name", "borough", "avg_speed", "trip_volume"]
        df = spark.createDataFrame(sample_data, columns)

        print("\n[TEST] Sample PySpark DataFrame:")
        df.show()

        # 5. GroupBy Aggregation
        print("[TEST] Executing PySpark GroupBy Aggregation:")
        agg_df = df.groupBy("borough").agg(
            count("location_id").alias("zone_count"),
            spark_round(avg("avg_speed"), 2).alias("mean_speed_mph")
        )
        agg_df.show()

        spark.stop()
        print("[VERIFIED] PySpark validation passed cleanly!")
        return 0

    except Exception as err:
        print(f"\n[ERROR] SparkSession execution failed: {err}")
        return 3

if __name__ == "__main__":
    sys.exit(main())
