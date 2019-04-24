#!/usr/bin/env python
# -*- coding:utf-8 -*-

import Sniper
task = Sniper.Task("TestGPU")
task.setLogLevel(2)

import MFSAlg
# task.property("svcs").append("")
task.property("algs").append("MFSAlg")

# ## the GoTask configuration
# go = task.createTask("Task/GoTask")
# go.property("svcs").append("FirstSvc")
# go.property("algs").append("SecondAlg")
# alg = go.find("SecondAlg")
# alg.property("SvcName").set("FirstSvc")

# ## the ChessTask configuration
# chess = task.createTask("Task/ChessTask")
# chess.property("svcs").append("FirstSvc")
# chess.property("algs").append("SecondAlg")
# alg = chess.find("SecondAlg")
# alg.property("SvcName").set("FirstSvc")

## ...
task.show()
task.setEvtMax(7)
task.run()
