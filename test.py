from io import StringIO  # Python 3
import sys

# Create the in-memory "file"
temp_out = StringIO()

# Replace default stdout (terminal) with our stream
sys.stdout = temp_out

print("This is going in to the memory stream")
temp_out.write("Can be written to like normal.\n")

fout = open('text.txt', 'w')

# The original `sys.stdout` is kept in a special
# dunder named `sys.__stdout__`. So you can restore
# the original output stream to the terminal.
sys.stdout = sys.__stdout__

print("Now printing back to terminal")
print("The 'fake' stdout contains: =======")
fout.write(temp_out.getvalue())
print("=============================")
