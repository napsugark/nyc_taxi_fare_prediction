import yaml
from pathlib import Path

def get_dvc_md5(dvc_file_path):
    with open(dvc_file_path, "r") as f:
        dvc_data = yaml.safe_load(f)
        return dvc_data.get("outs", [{}])[0].get("md5")