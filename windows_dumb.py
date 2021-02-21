import subprocess
import time
import sys
proc = subprocess.Popen("py -u batch_normalization.py", shell=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, universal_newlines=True)
f = open("text.txt", 'w+')
while True:
    rd = proc.stdout.readline()
    sys.stdout.write('\r'+rd.replace("\n", ""))
    sys.stdout.flush()
    if "[==============================]" in rd:
        print("\n\n")
        f.write(rd)
    if not rd:
        returncode = proc.poll()
        if returncode is not None:
            break
        time.sleep(0.1)
