import os
import subprocess as sub

run = "./build/visualHull" 
imagesAmount = 36
path = "./testData/kiwi/"
series = "kiwi"
pathlist = [x.path for x in os.scandir("./data/") if x.is_dir()]
print(pathlist )





if (len(pathlist) > 0):
    for i in pathlist:
        path = i
        series = i.split("/")[-1]
        print(path, series)
        runString = run + " " + str(imagesAmount) + " " + path + " " + series
        sub.run(runString, shell=True)
else:
    runString = run + " " + str(imagesAmount) + " " + path + " " + series
    sub.run(runString, shell=True)