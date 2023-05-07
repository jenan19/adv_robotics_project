import os
import subprocess as sub

run = "./build/visualHull" 
imagesAmount = 36
path = "./data/kiwi/"
series = "kiwi"

runString = run + " " + str(imagesAmount) + " " + path + " " + series
print(runString)
sub.run(runString, shell=True)