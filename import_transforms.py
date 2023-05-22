import csv
import copy
import numpy as np
import os

path = "/home/jc/adv_robotics_project/data/socket_dark/socket_dark_par.txt"

output_name  = "par.txt"

def a(text):
    chars = "[]' "
    for c in chars:
        text = text.replace(c, "")
    return text


out = []
with open(path, newline="\n") as file :
    reader = csv.reader(file, delimiter= ',')
    for r in reader:
        try:
            par_array = [2669.4684193141475, 0, 1308.7439664763986, 0, 2669.6106447899438, 1035.4419708519022, 0, 0, 1]
            array = [float(i) for i in a(str(r)).split(",")[:-4]]
            t = np.array([array.pop(3), array.pop(6), array.pop(9)])
            for i in t:
                array.append(i)
            for i in array:
                par_array.append(i)
            out.append(par_array)
        except:
            print("failed to read file...")
            break

with open(output_name, 'w') as f:
    for i, n in enumerate(out):
        f.write(str(i + 1) + " ")
        f.write(' '.join(map(str, np.around(n,decimals=10)))+ '\n')