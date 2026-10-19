#!/usr/bin/env python3
"""
TrafficPulse - MapReduce Validation Reducer
Aggregates key-value pairs (LocationID -> speed, distance, count)
Calculates mean speed, total volume, and average trip distance per zone.
"""
import sys

def main():
    current_loc = None
    total_speed = 0.0
    total_distance = 0.0
    trip_count = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            loc_id, metrics = line.split("\t", 1)
            speed_str, dist_str, count_str = metrics.split(",")
            speed = float(speed_str)
            dist = float(dist_str)
            count = int(count_str)
        except ValueError:
            continue

        if current_loc == loc_id:
            total_speed += speed
            total_distance += dist
            trip_count += count
        else:
            if current_loc is not None and trip_count > 0:
                avg_speed = total_speed / trip_count
                avg_dist = total_distance / trip_count
                print(f"{current_loc}\t{trip_count}\t{avg_speed:.2f}\t{avg_dist:.2f}")
            current_loc = loc_id
            total_speed = speed
            total_distance = dist
            trip_count = count

    if current_loc is not None and trip_count > 0:
        avg_speed = total_speed / trip_count
        avg_dist = total_distance / trip_count
        print(f"{current_loc}\t{trip_count}\t{avg_speed:.2f}\t{avg_dist:.2f}")

if __name__ == "__main__":
    main()
