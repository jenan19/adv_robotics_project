import os
import subprocess as sub
from tqdm import tqdm

run = "./build/visualHull" 
imagesAmount = 36
path = "./testData/kiwi/"
series = "kiwi"
pathlist = [x.path for x in os.scandir("./data/") if x.is_dir()]
#print(pathlist )

updateAll = True

plyFiles = [x.path for x in os.scandir("./plyFiles/") if x.is_dir()]


if (len(pathlist) > 0):
    for i in tqdm(pathlist):
        path = i
        series = i.split("/")[-1]
        #print(path, series)
        
        runString = run + " " + str(imagesAmount) + " " + path + "/" + " " + series

        if (os.path.isfile("./plyFiles/"+series+".ply") == False or updateAll == True):
            sub.run(runString, shell=True)
        else:
            print("./plyFiles/"+series+".ply does already exist.")

else:
    runString = run + " " + str(imagesAmount) + " " + path + " " + series
    sub.run(runString, shell=True)
