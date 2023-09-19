# Work in progress for faster training..
def trainInParallel(N):
  
    fns = []
    for i in range(N):
        fns.append(train)
    proc = []
    for fn in fns:
        p = mp.Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()



if(__name__=="__main__"):

    trainInParallel(2)
