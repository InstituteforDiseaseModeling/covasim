import os

# Compute the diff between the cached file and our results
stream = os.popen("diff -u tests/baselines/run_sim.json results/*.json")
diff = stream.read()
print(diff)

# Take off the --- and +++ lines denoting the file names (the first 2 lines)
without_file_names = diff.splitlines()[2:]

# Join by the chars \n which will be replaced later
# This allows us to pass around a multi-line string
diff = '\\n'.join(without_file_names)

# Set the output
print(f"::set-output name=JSON_OUTPUT::{diff}")
