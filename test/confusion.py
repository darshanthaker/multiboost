import math
import numpy as np

def main():
    newmatrix = []
    newfinal = []
    numclasses = 0
    oldmatrix = []
    oldfinal = []
    THRESHOLD = 0.065
    with open('OUTPUTOLD.txt') as inp:
        lineold= inp.readline()
        splitted = lineold.split('\t')
        while (not splitted[numclasses].endswith('\n')):
            numclasses += 1
        
        for i in range(0, numclasses):
            line = inp.readline()
            splitted = line.split('\t')
            splitted[numclasses] = splitted[numclasses][0:len(splitted[numclasses])-1]
            del splitted[0]
            #print splitted
            oldmatrix.append(splitted)

    for i in range(0, numclasses):
        tp = float(oldmatrix[i][i])
        fn = 0
        fp = 0
        tn = 0
        for row in range(0, len(oldmatrix)):
            for col in range(0, len(oldmatrix[0])):
                if (row == i and col != i):
                    fn += float(oldmatrix[row][col])
                elif (row != i and col == i):
                    fp += float(oldmatrix[row][col])
                elif (row != i and col != i):
                    tn += float(oldmatrix[row][col])
        oldfinal.append(list([tp, fn, fp, tn]))
    #print final

    with open('OUTPUT.txt') as inp:
        lineold= inp.readline()
        splitted = lineold.split('\t')
        while (not splitted[numclasses].endswith('\n')):
            numclasses += 1
        
        for i in range(0, numclasses):
            line = inp.readline()
            splitted = line.split('\t')
            splitted[numclasses] = splitted[numclasses][0:len(splitted[numclasses])-1]
            del splitted[0]
            #print splitted
            newmatrix.append(splitted)

    for i in range(0, numclasses):
        tp = float(newmatrix[i][i])
        fn = 0
        fp = 0
        tn = 0
        for row in range(0, len(newmatrix)):
            for col in range(0, len(newmatrix[0])):
                if (row == i and col != i):
                    fn += float(newmatrix[row][col])
                elif (row != i and col == i):
                    fp += float(newmatrix[row][col])
                elif (row != i and col != i):
                    tn += float(newmatrix[row][col])
        newfinal.append(list([tp, fn, fp, tn]))

    counter = 0
    for (oldconfusion, newconfusion) in zip(oldfinal, newfinal):
        #print "Current: " + str(counter)
        counter += 1
        oldtp = oldconfusion[0]
        oldfn = oldconfusion[1]
        oldfp = oldconfusion[2]
        oldtn = oldconfusion[3]

        if (oldtp == 0):
            continue
        if (oldfn == 0):
            continue
        if (oldfp == 0):
            continue
        if (oldtn == 0):
            continue
        
        oldtpr = oldtp/(oldtp + oldfn)
        oldtnr = oldtn/(oldtn + oldfp)
        oldppv = oldtp/(oldtp + oldfp)
        oldnpv = oldtn/(oldtn + oldfn)
        oldfpr = oldfp/(oldfp + oldtn)
        oldfnr = oldfn/(oldfn + oldtp)
        oldacc = (oldtp + oldtn)/(oldtp + oldfn + oldfp + oldtn)
        oldf1 = (2*oldtp)/(2*oldtp + oldfp + oldfn)
        oldmcc = ((oldtp*oldtn) - (oldfp*oldfn))/(math.sqrt((oldtp + oldfp)*(oldtp + oldfn)*(oldtn + oldfp)*(oldtn + oldfn)))


        newtp = newconfusion[0]
        newfn = newconfusion[1]
        newfp = newconfusion[2]
        newtn = newconfusion[3]
        if (newtp == 0):
            continue
        if (newfn == 0):
            continue
        if (newfp == 0):
            continue
        if (newtn == 0):
            continue
        
        newtpr = newtp/(newtp + newfn)
        newtnr = newtn/(newtn + newfp)
        newppv = newtp/(newtp + newfp)
        newnpv = newtn/(newtn + newfn)
        newfpr = newfp/(newfp + newtn)
        newfnr = newfn/(newfn + newtp)
        newacc = (newtp + newtn)/(newtp + newfn + newfp + newtn)
        newf1 = (2*newtp)/(2*newtp + newfp + newfn)
        newmcc = ((newtp*newtn) - (newfp*newfn))/(math.sqrt((newtp + newfp)*(newtp + newfn)*(newtn + newfp)*(newtn + newfn)))
        if (math.fabs(oldtpr - newtpr) > THRESHOLD):
            print "TPR is off: " + str(math.fabs(oldtpr - newtpr))
        if (math.fabs(oldtnr - newtnr) > THRESHOLD):
            print "TNR is off: " + str(math.fabs(oldtnr - newtnr))
        if (math.fabs(oldppv - newppv) > THRESHOLD):
            print "PPV is off: " + str(math.fabs(oldppv - newppv))
        if (math.fabs(oldnpv - newnpv) > THRESHOLD):
            print "NPV is off: " + str(math.fabs(oldnpv - newnpv))
        if (math.fabs(oldfpr - newfpr) > THRESHOLD):
            print "FPR is off: " + str(math.fabs(oldfpr - newfpr))
        if (math.fabs(oldfnr - newfnr) > THRESHOLD):
            print "FNR is off: " + str(math.fabs(oldfnr - newfnr))
        if (math.fabs(oldacc - newacc) > THRESHOLD):
            print "ACC is off: " + str(math.fabs(oldacc - newacc))
        if (math.fabs(oldf1 - newf1) > THRESHOLD):
            print "F1 is off: " + str(math.fabs(oldf1 - newf1))
        if (math.fabs(oldmcc - newmcc) > THRESHOLD):
            print "MCC is off: " + str(math.fabs(oldmcc - newmcc))
        print "oldacc: " + str(oldacc)
        print "newacc: " + str(newacc)
        """
        print "oldtpr: " + str(oldtpr)
        print "oldtnr: " + str(oldtnr)
        print "oldppv: " + str(oldppv)
        print "oldnpv: " + str(oldnpv)
        print "oldfpr: " + str(oldfpr)
        print "oldfnr: " + str(oldfnr)
        print "oldacc: " + str(oldacc)
        print "oldf1: " + str(oldf1)
        print "oldmcc: " + str(oldmcc)


        print "newtpr: " + str(newtpr)
        print "newtnr: " + str(newtnr)
        print "newppv: " + str(newppv)
        print "newnpv: " + str(newnpv)
        print "newfpr: " + str(newfpr)
        print "newfnr: " + str(newfnr)
        print "newacc: " + str(newacc)
        print "newf1: " + str(newf1)
        print "newmcc: " + str(newmcc)
        """

    print "All done comparing confusion matrices"
        
        
        
        
        
                    
                    
        
         

if __name__=='__main__':
    main()
