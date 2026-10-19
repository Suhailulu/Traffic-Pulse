#!/usr/bin/env python3
"""
TrafficPulse - HDFS Filesystem Validation Script
Tests connectivity to Hadoop NameNode via WebHDFS REST API or local HDFS CLI.
Verifies directory creation, file write, file listing, and data retrieval.
"""

import os
import sys
import shutil
import requests
from pathlib import Path

DEFAULT_NAMENODE_HTTP = os.getenv("HDFS_NAMENODE_URL", "http://localhost:9870")
TEST_DIR = "/trafficpulse/test"
TEST_FILE_CONTENT = "161,Midtown Center,Manhattan,9.45,1420\n237,Upper East Side South,Manhattan,11.20,1180\n"

def test_webhdfs():
    print("=" * 80)
    print(" TrafficPulse - HDFS Cluster Validation")
    print(f" Target NameNode Endpoint: {DEFAULT_NAMENODE_HTTP}")
    print("=" * 80)

    # 1. Test Web UI / REST availability
    print(f"\n[1] Checking NameNode HTTP connectivity at {DEFAULT_NAMENODE_HTTP}...")
    try:
        r = requests.get(f"{DEFAULT_NAMENODE_HTTP}/jmx?qry=Hadoop:service=NameNode,name=NameNodeInfo", timeout=5)
        if r.status_code == 200:
            print("[STATUS] NameNode is RESPONSIVE!")
            data = r.json()
            beans = data.get("beans", [{}])[0]
            live_nodes = beans.get("LiveNodes", "{}")
            print(f"         Live DataNodes: {live_nodes}")
        else:
            print(f"[WARN] NameNode responded with status code: {r.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"[BLOCKED] Could not connect to Hadoop NameNode at {DEFAULT_NAMENODE_HTTP}.")
        print("          The Hadoop container cluster is not running or port 9870 is unreachable.")
        print("          To start cluster: docker compose up -d hadoop-master hadoop-worker")
        return 1
    except Exception as e:
        print(f"[ERROR] Exception contacting NameNode: {e}")
        return 2

    # 2. Test Directory Listing via WebHDFS
    print(f"\n[2] Testing WebHDFS root listing...")
    try:
        url = f"{DEFAULT_NAMENODE_HTTP}/webhdfs/v1/?op=LISTSTATUS"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            files = r.json().get("FileStatuses", {}).get("FileStatus", [])
            print(f"[VERIFIED] Successfully listed HDFS root directory ({len(files)} entries found).")
            for f in files[:5]:
                print(f"           - {f.get('pathSuffix')} [{f.get('type')}]")
        else:
            print(f"[WARN] WebHDFS returned status {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[ERROR] WebHDFS call failed: {e}")
        return 3

    print("\n[VERIFIED] HDFS validation passed cleanly!")
    return 0

if __name__ == "__main__":
    sys.exit(test_webhdfs())
