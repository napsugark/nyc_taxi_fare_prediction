from dotenv import load_dotenv
import os
import subprocess

load_dotenv()  # loads AWS credentials

def run_pipeline():
    env = os.environ.copy()  # Get current env, including loaded AWS credentials
    subprocess.run(["dvc", "repro"], check=True, env=env)
    subprocess.run(["dvc", "push"], check=True, env=env)
    subprocess.run(["git", "add", "."], check=True, env=env)
    subprocess.run(["git", "commit", "-m", "auto-run"], check=True, env=env)

    
def main():
    run_pipeline()

if __name__ == "__main__":
    main()