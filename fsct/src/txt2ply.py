import os
import sys
import ply_io
import numpy as np
import pandas

input_name = os.path.basename(sys.argv[1])
input_name = os.path.splitext(input_name)[0]
dir_name = os.path.dirname(sys.argv[1])

xyz = np.genfromtxt(sys.argv[1])
ply_io.write_ply(os.path.join(dir_name, input_name + '.ply'), pandas.DataFrame(xyz[:, :4], columns = ['x','y','z','lw']))

