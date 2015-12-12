#!/bin/bash

cd ~/multiboost/build
cmake ../ >/dev/null
make >/dev/null 2>&1
cd ~/multiboostOLD/build
cmake ../ >/dev/null
make >/dev/null 2>&1

cd ~/multiboost/test


NEWTRAIN=$(~/multiboost/build/multiboost --stronglearner AdaBoostPL --nworkers 7 --nworkers2 3 --fileformat arff --train pendigitsTrain.arff 200 --verbose 0 --learnertype SingleStumpLearner --outputinfo resultsSingleStump.dta --shypname shypSingleStump.xml | grep "Training time" )

NEWTEST=$(~/multiboost/build/multiboost --stronglearner AdaBoostPL --nworkers 7 --nworkers2 3 --fileformat arff --cmatrix pendigitsTest.arff shypSingleStump0.xml OUTPUT.txt | grep "Testing time")

TRAIN=$(echo $NEWTRAIN | awk  '{print $NF}')
TEST=$(echo $NEWTEST | awk '{print $NF}')
echo "NEW TIMES:"
python -c "print $TRAIN + $TEST"


OLDTRAIN=$(~/multiboostOLD/build/multiboost --fileformat arff --train pendigitsTrain.arff 200 --verbose 0 --learnertype SingleStumpLearner --outputinfo resultsSingleStumpOLD.dta --shypname shypSingleStumpOLD.xml | grep "Training time")
OLDTEST=$(~/multiboostOLD/build/multiboost --fileformat arff --cmatrix pendigitsTest.arff shypSingleStumpOLD.xml OUTPUTOLD.txt | grep "Testing time")

TRAIN=$(echo $OLDTRAIN | awk '{print $NF}')
TEST=$(echo $OLDTEST | awk '{print $NF}')
echo "OLD TIMES:"
python -c "print $TRAIN + $TEST"

diff -s OUTPUT.txt OUTPUTOLD.txt
python confusion.py
