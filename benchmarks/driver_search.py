import sys
import subprocess

# Define the command to run the Python script
# 'python' command might need to be replaced with 'python3' on some systems
command = [sys.executable, 'benchmarks/test.py']

# Run the command
process = subprocess.run(command, capture_output=True, text=True)

# Check if the process was successful
if process.returncode == 0:
    print('Success:')
    print(process.stdout)  # Print standard output of the command
else:
    print(process.stdout)  # Print standard error of the command
    print('Error:')
    print(process.stderr)  # Print standard error of the command
