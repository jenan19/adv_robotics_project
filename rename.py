import os

DIR = os.getcwd() + "/debugs/debug_light"
cnt = 1
for file_ in os.listdir(DIR):
    if file_.split('.')[-1] != "txt":
        os.rename(DIR + "/" + file_,DIR + "/" +  "socket_light" + str(cnt) + ".png")
        cnt += 1
