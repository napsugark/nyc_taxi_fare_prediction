from dotenv import load_dotenv
import os
import subprocess
import argparse

load_dotenv()  # loads AWS credentials

def run_pipeline(commit_message):
    env = os.environ.copy()  # Get current env, including loaded AWS credentials
    subprocess.run(["dvc", "repro"], check=True, env=env)
    subprocess.run(["dvc", "push"], check=True, env=env)
    subprocess.run(["git", "add", "."], check=True, env=env)
    subprocess.run(["git", "commit", "-m", commit_message], check=True, env=env)

    
def main():
    parser = argparse.ArgumentParser(description="Run DVC pipeline and push results.")
    parser.add_argument(
        "-m", "--message",
        type=str,
        default="auto-run",
        help="Git commit message"
    )
    args = parser.parse_args()
    run_pipeline(args.message)

if __name__ == "__main__":
    main()