import os
import subprocess as sub
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
from multiprocessing import Process

run = "./build/visualHull" 
imagesAmount = 36
path = "./testData/kiwi/"
series = "kiwi"
pathlist = [x.path for x in os.scandir("./data/") if x.is_dir()]
#print(pathlist )

updateAll = True

plyFiles = [x.path for x in os.scandir("./plyFiles/") if x.is_dir()]




def runCpp( series):

    runString = run + " " + str(imagesAmount) + " " + path + "/" + " " + series

    if (os.path.isfile("./plyFiles/"+series+".ply") == False or updateAll == True):
        sub.run(runString, shell=True)
    else:
        print("./plyFiles/"+series+".ply does already exist.")
    

if (len(pathlist) > 0):
    seriesList = []
    for i in pathlist:
        path = i
        series = i.split("/")[-1]
        seriesList.append(series)

    
    with Pool() as pool:
        pool.map(runCpp,seriesList)

    # # create a thread pool
    # with ThreadPool() as pool:
    #     # call the function for each item concurrently
    #     pool.map(runCpp, seriesList)

    #pool.close()
else:
    runString = run + " " + str(imagesAmount) + " " + path + " " + series
    sub.run(runString, shell=True)
# close the thread pool

