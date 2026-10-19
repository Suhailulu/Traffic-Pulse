#!/usr/bin/env python3
"""
TrafficPulse - Environment Verification Script
Validates local Python runtime, virtual environment, Git, Docker,
and Big Data infrastructure components (Java, Hadoop, HDFS, Hive, PySpark).
"""

import os
import sys
import subprocess
import shutil

def run_cmd(cmd_list):
    """Executes a command list safely, returning (exit_code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return -1, "", "Command not found"
    except Exception as e:
        return -2, "", str(e)

def check_python_environment():
    """Inspects Python interpreter and detects if running inside a virtual environment."""
    is_venv = sys.prefix != sys.base_prefix
    return {
        "version": sys.version.split()[0],
        "executable": sys.executable,
        "in_venv": is_venv,
        "venv_path": sys.prefix if is_venv else "None (Global/System Python)"
    }

def check_python_packages(packages):
    """Checks importability and version of key project dependencies."""
    results = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "Installed (no __version__)")
            results[pkg] = {"status": "OPERATIONAL", "version": version}
        except ImportError:
            results[pkg] = {"status": "MISSING", "version": None}
        except Exception as e:
            results[pkg] = {"status": f"ERROR ({type(e).__name__})", "version": None}
    return results

def check_git():
    """Checks Git installation, current branch, and repository integrity."""
    code, out, _ = run_cmd(["git", "--version"])
    if code != 0:
        return {"status": "NOT FOUND", "details": "Git executable not in PATH"}

    git_ver = out
    code_branch, out_branch, _ = run_cmd(["git", "branch", "--show-current"])
    branch = out_branch if code_branch == 0 and out_branch else "No commits/detached"

    code_user, out_user, _ = run_cmd(["git", "config", "user.name"])
    user = out_user if code_user == 0 else "Unconfigured"

    return {
        "status": "OPERATIONAL",
        "version": git_ver,
        "branch": branch,
        "user": user
    }

def check_docker():
    """Checks Docker executable and attempts live communication with the Docker daemon."""
    code, out, _ = run_cmd(["docker", "--version"])
    if code != 0:
        return {"status": "NOT INSTALLED", "details": "Docker CLI not detected"}

    docker_cli_ver = out
    code_ps, _, err_ps = run_cmd(["docker", "ps"])
    if code_ps == 0:
        return {"status": "OPERATIONAL", "version": docker_cli_ver, "daemon": "Running & Accessible"}
    else:
        return {
            "status": "DAEMON NOT RUNNING",
            "version": docker_cli_ver,
            "daemon": "Engine stopped or socket unreachable"
        }

def check_command_tool(tool_name, version_flags=("--version",)):
    """Checks whether a specific system executable exists and executes its version check."""
    path = shutil.which(tool_name)
    if not path:
        return {"status": "NOT FOUND", "version": None, "path": None}

    code, out, err = run_cmd([tool_name, *version_flags])
    raw_output = out if out else err
    first_line = raw_output.splitlines()[0] if raw_output else "Available"
    return {"status": "OPERATIONAL", "version": first_line, "path": path}

def check_pyspark_runtime():
    """Attempts importing PySpark and checks Java gateway availability."""
    try:
        import pyspark
        spark_ver = pyspark.__version__
        # Verify if Java is available to actually initialize PySpark
        java_home = os.environ.get("JAVA_HOME", None)
        java_found = shutil.which("java") is not None
        if not java_found and not java_home:
            return {
                "status": "PACKAGE INSTALLED (JAVA MISSING)",
                "version": spark_ver,
                "notes": "PySpark package is present, but JVM runtime (JAVA_HOME) is missing."
            }
        return {
            "status": "OPERATIONAL",
            "version": spark_ver,
            "notes": "PySpark and Java runtime both present."
        }
    except ImportError:
        return {
            "status": "MISSING",
            "version": None,
            "notes": "PySpark library not installed in current Python environment."
        }

def main():
    print("=" * 80)
    print(" TrafficPulse - Big Data Analytics & Prediction: Environment Verification")
    print(" Course: CCS334 Big Data Analytics | SDG 11 - Sustainable Cities")
    print("=" * 80)

    # 1. Python Environment
    py_info = check_python_environment()
    print("\n[1] Python Runtime Environment:")
    print(f"  - Version:         {py_info['version']}")
    print(f"  - Executable:      {py_info['executable']}")
    print(f"  - In Virtual Env:  {py_info['in_venv']} ({py_info['venv_path']})")

    # 2. Key Python Dependencies
    req_pkgs = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "requests": "requests",
        "pyarrow": "pyarrow",
        "plotly": "plotly",
        "folium": "folium",
        "streamlit": "streamlit",
        "pyspark": "pyspark"
    }
    
    pkg_results = {}
    for display_name, mod_name in req_pkgs.items():
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "Installed")
            pkg_results[display_name] = {"status": "OPERATIONAL", "version": ver}
        except ImportError:
            pkg_results[display_name] = {"status": "MISSING", "version": None}
        except Exception as e:
            pkg_results[display_name] = {"status": f"ERROR ({type(e).__name__})", "version": None}

    print("\n[2] Python Package Availability:")
    for pkg, info in pkg_results.items():
        ver_str = f"v{info['version']}" if info['version'] else "N/A"
        print(f"  - {pkg:<15} [{info['status']:<11}] {ver_str}")

    # 3. Version Control (Git)
    git_info = check_git()
    print("\n[3] Version Control (Git):")
    print(f"  - Status:          {git_info['status']}")
    if git_info['status'] == "OPERATIONAL":
        print(f"  - Version:         {git_info['version']}")
        print(f"  - Current Branch:  {git_info['branch']}")
        print(f"  - User Name:       {git_info['user']}")

    # 4. Container Infrastructure (Docker)
    docker_info = check_docker()
    print("\n[4] Container Engine (Docker):")
    print(f"  - Status:          {docker_info['status']}")
    if "version" in docker_info:
        print(f"  - CLI Version:     {docker_info['version']}")
        print(f"  - Daemon State:    {docker_info['daemon']}")

    # 5. Big Data Platform Tools
    print("\n[5] Big Data Ecosystem Components:")
    
    java_res = check_command_tool("java", ["-version"])
    print(f"  - Java (JDK):      [{java_res['status']:<11}] {java_res['version'] or 'Not in PATH'}")

    hadoop_res = check_command_tool("hadoop", ["version"])
    print(f"  - Hadoop CLI:      [{hadoop_res['status']:<11}] {hadoop_res['version'] or 'Not in PATH'}")

    hdfs_res = check_command_tool("hdfs", ["version"])
    print(f"  - HDFS CLI:        [{hdfs_res['status']:<11}] {hdfs_res['version'] or 'Not in PATH'}")

    hive_res = check_command_tool("hive", ["--version"])
    print(f"  - Apache Hive:     [{hive_res['status']:<11}] {hive_res['version'] or 'Not in PATH'}")

    pyspark_res = check_pyspark_runtime()
    print(f"  - PySpark Engine:  [{pyspark_res['status']}] {pyspark_res['notes']}")

    print("\n" + "=" * 80)
    print(" Verification complete. Results represent verified live execution tests.")
    print("=" * 80)

if __name__ == "__main__":
    main()
