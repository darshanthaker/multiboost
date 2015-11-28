#!/bin/bash

cd ~/multiboost/build
cmake ../
make
cd ~/multiboostOLD/build
cmake ../
make

cd ~/multiboost/test

~/multiboost/build/multiboost --stronglearner AdaBoostPL --nworkers 1 --fileformat arff --train pendigitsTrain.arff 1 --verbose 3 --learnertype SingleStumpLearner --outputinfo resultsSingleStump.dta --shypname shypSingleStump.xml 
~/multiboost/build/multiboost --stronglearner AdaBoostPL --nworkers 1 --fileformat arff --cmatrix pendigitsTest.arff shypSingleStump.xml OUTPUT.txt

~/multiboostOLD/build/multiboost --fileformat arff --train pendigitsTrain.arff 1 --verbose 3 --learnertype SingleStumpLearner --outputinfo resultsSingleStumpOLD.dta --shypname shypSingleStumpOLD.xml
~/multiboostOLD/build/multiboost --fileformat arff --cmatrix pendigitsTest.arff shypSingleStumpOLD.xml OUTPUTOLD.txt

diff -s OUTPUT.txt OUTPUTOLD.txt
# TODO: Python confusion matrix??
