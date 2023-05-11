import os

DIR = os.getcwd() + "/debugs/debug_light"
for idx, file_ in enumerate(os.listdir(DIR)):
    os.rename(DIR + "/" + file_,DIR + "/" +  "socket_light" + str(idx) + ".png")
