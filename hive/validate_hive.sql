-- ==============================================================================
-- TrafficPulse - Apache Hive Metastore & HiveServer2 Validation Script
-- Validates database creation, table creation, sample record insertion & HiveQL aggregations
-- ==============================================================================

-- 1. Create dedicated verification database
CREATE DATABASE IF NOT EXISTS trafficpulse_validation;
USE trafficpulse_validation;

-- 2. Create small sample traffic zone metrics table
DROP TABLE IF EXISTS sample_zone_traffic;
CREATE TABLE sample_zone_traffic (
    location_id INT,
    zone_name STRING,
    borough STRING,
    avg_speed DOUBLE,
    trip_volume INT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE;

-- 3. Insert verified sample rows
INSERT INTO sample_zone_traffic VALUES
    (161, 'Midtown Center', 'Manhattan', 9.45, 1420),
    (237, 'Upper East Side South', 'Manhattan', 11.20, 1180),
    (132, 'JFK Airport', 'Queens', 24.80, 890),
    (138, 'LaGuardia Airport', 'Queens', 21.30, 940),
    (79, 'East Village', 'Manhattan', 10.15, 760);

-- 4. Analytical verification query: Borough level speed and volume aggregation
SELECT 
    borough,
    COUNT(*) as total_zones,
    ROUND(AVG(avg_speed), 2) as borough_mean_speed_mph,
    SUM(trip_volume) as borough_total_trips
FROM sample_zone_traffic
GROUP BY borough
ORDER BY borough_total_trips DESC;
