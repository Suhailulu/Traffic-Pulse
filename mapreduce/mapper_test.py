#!/usr/bin/env python3
"""
TrafficPulse - MapReduce Validation Mapper
Reads comma- or tab-separated records of (LocationID, Speed_mph, Trip_Distance)
Emits: LocationID\tSpeed_mph,Trip_Distance,1
"""
import sys

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("LocationID"):
            continue
        parts = line.split(",")
        if len(parts) >= 3:
            loc_id = parts[0].strip()
            try:
                speed = float(parts[1].strip())
                dist = float(parts[2].strip())
                print(f"{loc_id}\t{speed:.2f},{dist:.2f},1")
            except ValueError:
                continue

if __name__ == "__main__":
    main()
