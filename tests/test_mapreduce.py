"""
TrafficPulse - Hadoop MapReduce Unit Tests
Tests mapper key-value emission, invalid record filtering, and reducer aggregation logic.
"""

import io
import sys
import pytest

from mapreduce.mapper_test import main as mapper_main
from mapreduce.reducer_test import main as reducer_main

def test_mapper_valid_record_emission(monkeypatch, capsys):
    """Verifies that mapper correctly parses CSV lines and emits tab-separated key-values."""
    input_data = (
        "LocationID,Speed_mph,Trip_Distance\n"
        "161,12.50,2.40\n"
        "237,18.00,4.00\n"
        "# Comment line\n"
        "invalid,data\n"
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(input_data))
    mapper_main()

    captured = capsys.readouterr().out.strip().splitlines()
    assert len(captured) == 2
    assert captured[0] == "161\t12.50,2.40,1"
    assert captured[1] == "237\t18.00,4.00,1"

def test_reducer_aggregation(monkeypatch, capsys):
    """Verifies that reducer aggregates sorted mapper outputs by key."""
    # Sorted input (as produced by Hadoop shuffle/sort phase)
    sorted_mapper_output = (
        "161\t10.00,2.00,1\n"
        "161\t14.00,4.00,1\n"
        "237\t20.00,5.00,1\n"
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(sorted_mapper_output))
    reducer_main()

    captured = capsys.readouterr().out.strip().splitlines()
    assert len(captured) == 2

    # Zone 161: count=2, avg_speed=12.00, avg_dist=3.00
    parts_161 = captured[0].split("\t")
    assert parts_161[0] == "161"
    assert parts_161[1] == "2"
    assert float(parts_161[2]) == 12.00
    assert float(parts_161[3]) == 3.00

    # Zone 237: count=1, avg_speed=20.00, avg_dist=5.00
    parts_237 = captured[1].split("\t")
    assert parts_237[0] == "237"
    assert parts_237[1] == "1"
    assert float(parts_237[2]) == 20.00
    assert float(parts_237[3]) == 5.00

def test_end_to_end_streaming_pipeline(monkeypatch, capsys):
    """Simulates Hadoop Streaming: raw CSV -> mapper -> sort -> reducer."""
    raw_csv = (
        "161,15.00,3.00\n"
        "161,10.00,2.00\n"
        "161,5.00,1.00\n"
    )
    # 1. Run Mapper
    monkeypatch.setattr("sys.stdin", io.StringIO(raw_csv))
    mapper_main()
    mapper_out = capsys.readouterr().out

    # 2. Simulate Hadoop Shuffle/Sort
    sorted_lines = sorted(mapper_out.strip().splitlines())
    sorted_input = "\n".join(sorted_lines) + "\n"

    # 3. Run Reducer
    monkeypatch.setattr("sys.stdin", io.StringIO(sorted_input))
    reducer_main()
    reducer_out = capsys.readouterr().out.strip().splitlines()

    assert len(reducer_out) == 1
    loc_id, count, avg_spd, avg_dst = reducer_out[0].split("\t")
    assert loc_id == "161"
    assert count == "3"
    assert float(avg_spd) == 10.00  # (15 + 10 + 5) / 3 = 10.0
    assert float(avg_dst) == 2.00   # (3 + 2 + 1) / 3 = 2.0
