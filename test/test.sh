#!/bin/bash

cd ~/multiboost/build
cmake ../
make
cd ~/multiboostOLD/build
cmake ../
make

cd ~/multiboost/test

~/multiboost/build/multiboost --stronglearner AdaBoostPL --nworkers 7 --fileformat arff --train pendigitsTrain.arff 200 --verbose 0 --learnertype SingleStumpLearner --outputinfo resultsSingleStump.dta --shypname shypSingleStump.xml 

~/multiboost/build/multiboost --stronglearner AdaBoostPL --nworkers 7 --fileformat arff --cmatrix pendigitsTest.arff shypSingleStump0.xml OUTPUT.txt

echo "-------------------------------------------------------------"

~/multiboostOLD/build/multiboost --fileformat arff --train pendigitsTrain.arff 200 --verbose 0 --learnertype SingleStumpLearner --outputinfo resultsSingleStumpOLD.dta --shypname shypSingleStumpOLD.xml
~/multiboostOLD/build/multiboost --fileformat arff --cmatrix pendigitsTest.arff shypSingleStumpOLD.xml OUTPUTOLD.txt

diff -s OUTPUT.txt OUTPUTOLD.txt
# TODO: Python confusion matrix??
