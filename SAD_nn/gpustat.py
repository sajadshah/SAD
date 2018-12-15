import subprocess, sys, getopt, os, time

def runProcess(exe):
    p = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while(True):
      retcode = p.poll() #returns None while subprocess is running
      line = p.stdout.readline()
      yield line
      if(retcode is not None):
        break

def getTheanoFlags():

    cmd = "./gpustat.sh"
    print(cmd)
    # p = subprocess.Popen([cmd], stdout=subprocess.PIPE, stderr = subprocess.PIPE)
    out = ""
    for line in runProcess(cmd.split()):
        out += line
    freeMemsStr = out.split(',')
    print(freeMemsStr)
    freeMems = []
    index = 0
    minim = 10000000
    for i in range(len(freeMemsStr)):
        s = freeMemsStr[i]
        t = s.split()
        x = int(t[0])
        if( x < minim ):
            index = i
        freeMems.append(x)

    result = "THEANO_FLAGS='device=gpu" + str(index) + "'"
    return result


