#!/bin/bash

cd ~/multiboost/build
make
cd ~/multiboostOLD/build
make

cd ~/multiboost/test

~/multiboost/build/multiboost --fileformat arff --train pendigitsTrain.arff 1 --verbose 3 --learnertype SingleStumpLearner --outputinfo resultsSingleStump.dta --shypname shypSingleStump.xml
~/multiboost/build/multiboost --fileformat arff --cmatrix pendigitsTest.arff shypSingleStump.xml OUTPUT.txt

~/multiboostOLD/build/multiboost --fileformat arff --train pendigitsTrain.arff 1 --verbose 3 --learnertype SingleStumpLearner --outputinfo resultsSingleStumpOLD.dta --shypname shypSingleStumpOLD.xml
~/multiboostOLD/build/multiboost --fileformat arff --cmatrix pendigitsTest.arff shypSingleStumpOLD.xml OUTPUTOLD.txt

diff OUTPUT.txt OUTPUTOLD.txt
