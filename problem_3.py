#!/usr/bin/env python
#-*- coding: utf8 -*-

import os, numpy as np, sys, matplotlib.pyplot as plt

reload(sys)
sys.setdefaultencoding("utf-8")

x = []
y = []
z = []
t1 = []
t2 = []

truth_file = "sift_groundtruth.txt"
ans = np.loadtxt(truth_file)
config = {
    "-n": 10,
    "-d": 10000,
    "-q": 10,
    "-I": "/Users/summer/Downloads/sift/sift_base.fvecs",
    "-Q": "/Users/summer/Downloads/sift/sift_query.fvecs",
    "-O": "/Users/summer/Downloads/result.txt"
}


# stripe is 10
for i in range(1, config["-d"], 10):
    config["-d"] = i
    comd = "./main "

    for item in config:
        comd += str(item) + " " + str(config[item]) + " "

    time = os.popen(comd).readline().strip().split(' ')

    t1.append(int(time[0]))
    t2.append(int(time[1]))

    lshRet = np.loadtxt(config["-O"])
    kdtree = np.loadtxt(config["-O"] + ".2")

    lshrate = 0.0
    kdrate = 0.0

    q_num = config["-q"]
    for j in range(q_num):
        tmp = ans[len(ans) - j - 1]
        lshrate += len(set(set(tmp) & set(lshRet[j]))) / 100.0
        kdrate += len(set(set(tmp) & set(kdtree[j]))) / 100.0

    lshrate /= q_num*1.0
    kdrate /= q_num*1.0

    x.append(i)
    y.append(lshrate)
    z.append(kdrate)

x = np.array(x)
y = np.array(y)
z = np.array(z)

plt.figure(1)
plt.figure(2)

plt.figure(1)
plt.title("Accuracy")
line1 = plt.plot(x,y)
line2 = plt.plot(x,z)
plt.ylabel("Accuracy")
plt.xlabel("data size")
plt.setp(line1, color = "r", linewidth=2)
plt.setp(line2, color = "g", linewidth=2)


plt.figure(2)
plt.title("Times")
line1 = plt.plot(x,t1)
line2 = plt.plot(x,t2)
plt.ylabel("clock times")
plt.xlabel("data size")
plt.setp(line1, color="b", linewidth=2)
plt.setp(line2, color="r", linewidth=2)

plt.show()







