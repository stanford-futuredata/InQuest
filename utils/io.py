"""
Utility functions to help with writing and reading data and results.
"""
import boto3
import json
import os


def print_results(results_df, config) -> None:
    """
    Print results in results_df in a formatted way.
    """
    pass


def write_json(data, ts, local, results_dir=None):
    """
    Write JSON data to local directory or to s3.
    """
    filename = f"config_{ts}.json"

    # if local is True, write data to local filepath
    if local:
        filepath = f"{os.path.join(results_dir, filename)}"
        with open(filepath, 'w') as f:
            json.dump(data, f)

    # if local is False, write data to s3
    else:
        s3_key = f"sim-results/{filename}"

        s3 = boto3.resource("s3")
        s3_object = s3.Object("wildfires-data", s3_key)
        s3_object.put(Body=json.dumps(data).encode("utf-8"))
