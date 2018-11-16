#!/usr/bin/env python
# -*- coding:utf-8 -*-

import Sniper
task = Sniper.Task("TestGPU")
task.setEvtMax(5)
task.setLogLevel(2) ## Try a different LogLevel

import SamplingSvc
import TestAlg

#########################################
task.property("svcs").append("SamplingSvc")
task.property("algs").append("TestAlg")
alg = task.find("TestAlg")
alg.property("Sampling").set("SamplingSvc")

#########################################
## We can customize the instance names of the algorithm and service
## So that we can create more than one instances for each algorithm/service type
#task.property("svcs").append("FirstSvc/SpikeDog")
#task.property("algs").append("SecondAlg/TomCat")
#task.property("algs").append("SecondAlg/JerryMouse")
#alg = task.find("TomCat")
#alg.property("SvcName").set("SpikeDog")
#alg = task.find("JerryMouse")
#alg.property("SvcName").set("SpikeDog")

#########################################
task.show()
task.run()
