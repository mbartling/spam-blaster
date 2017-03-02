import sys
import glob
import os
import re

mFiles = glob.glob(sys.argv[1])
targetDir = os.path.dirname(sys.argv[1]) + "_split"

for mFile in mFiles:
    baseName = os.path.splitext(os.path.basename(mFile))[0]
    with open(mFile) as fp:
        data = fp.read()
        data = re.split("[.\n]", data)
        for i, line in enumerate(data):
            with open(os.path.join(targetDir, "%s_P%d.txt" % (baseName, i)), "w") as op:
                op.write(line)

