import os
import subprocess as sub

runString = "./build/visualHull" 
imagesAmountString = 36
pathString = "./data/kiwi/"
seriesString = "kiwi"

runString = runString + " " + str(imagesAmountString) + " " + pathString + " " + seriesString
print(runString)
sub.run(runString, shell=True)