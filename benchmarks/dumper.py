import os
import pickle
import tempfile
import time

import random
import numpy as np

# asm (add CuAsm to PYTHONPATH!)
from CuAsm.CubinFile import CubinFile

from absl import app
from absl import flags

# YAPF: disable
FLAGS = flags.FLAGS

# kernel
flags.DEFINE_string("f", None, "")
flags.DEFINE_string("save", None, "")

def main(_):

    with open(FLAGS.f, 'rb') as f:
        data = pickle.load(f)
    cubin = data['cubin']

    with tempfile.NamedTemporaryFile(mode='wb', delete=True) as temp_file:

        temp_file.write(cubin)
        # Ensure data is written to the file before reading it
        temp_file.flush()
        temp_file.seek(0)

        time.sleep(1)
        cf = CubinFile(temp_file.name)

    cf.saveAsCuAsm(FLAGS.save+'.cuasm')

if __name__ == "__main__":
    app.run(main)
