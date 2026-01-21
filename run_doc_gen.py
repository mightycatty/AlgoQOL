# ... existing code ...
import subprocess
import sys

# Define the run_script function
def run_script():
    try:
        result = subprocess.run([sys.executable, 'tools/gen_api_docs.py'], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except Exception as e:
        print(f"Failed to run script: {e}")

if __name__ == "__main__":
    run_script()
