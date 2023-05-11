import csv
import copy
import numpy as np
path = "transforms.txt"
with open(path, newline="\n") as file :
    with open("par.txt", 'w', newline='\n') as output:
        reader = csv.reader(file, delimiter= ',')
        writer = csv.writer(output)
        for row in reader:
            new = copy.copy(row)
            for i, element in enumerate(row):
                if i != 3 and i != 6 and i != 9:
                    new[i] = element
            new[-3] = row[3]
            new[-2] = row[6]
            new[-1] = row[9]
            
            
            print(new)
            
            writer.writerow(new)