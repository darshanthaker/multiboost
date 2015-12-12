#!/bin/bash

cd ~/multiboost/build
cmake ../ 
make
cd ~/multiboostOLD/build
cmake ../ 
make 
echo "Finished compiling"

cd ~/multiboost/test

INPUT=('dexter' 'yeast' 'pendigits')
INPUT_COUNTER=0
INPUT_SIZE=3
WORKER_COUNTER=1
WORKER_SIZE=17

while [ $INPUT_COUNTER -lt $INPUT_SIZE ]; do
    while [ $WORKER_COUNTER -lt $WORKER_SIZE ]; do
        echo "Thread count: $WORKER_COUNTER, Input: ${INPUT[$INPUT_COUNTER]}"
        NEWTRAIN=$(~/multiboost/build/multiboost --stronglearner AdaBoostPL --nworkers $WORKER_COUNTER --nworkers2 $WORKER_COUNTER --fileformat arff --train ${INPUT[$INPUT_COUNTER]}train.arff 200 --verbose 0 --learnertype SingleStumpLearner --outputinfo resultsSingleStump.dta --shypname shypSingleStump.xml | grep "Training time")

        NEWTEST=$(~/multiboost/build/multiboost --stronglearner AdaBoostPL --nworkers $WORKER_COUNTER --nworkers2 $WORKER_COUNTER --fileformat arff --cmatrix ${INPUT[$INPUT_COUNTER]}test.arff shypSingleStump0.xml OUTPUT.txt  | grep "Testing time")

        TRAIN=$(echo $NEWTRAIN | awk  '{print $NF}')
        TEST=$(echo $NEWTEST | awk '{print $NF}')
        #echo "HERE THEY COME"
        #echo $TRAIN
        #echo $TEST


        OLDTRAIN=$(~/multiboostOLD/build/multiboost --fileformat arff --train ${INPUT[$INPUT_COUNTER]}train.arff 200 --verbose 0 --learnertype SingleStumpLearner --outputinfo resultsSingleStumpOLD.dta --shypname shypSingleStumpOLD.xml | grep "Training time")
        OLDTEST=$(~/multiboostOLD/build/multiboost --fileformat arff --cmatrix ${INPUT[$INPUT_COUNTER]}test.arff shypSingleStumpOLD.xml OUTPUTOLD.txt | grep "Testing time")

        OLDTTRAIN=$(echo $OLDTRAIN | awk '{print $NF}')
        OLDTTEST=$(echo $OLDTEST | awk '{print $NF}')
        #python -c "print $TRAIN + $TEST"
        python -c "print $OLDTTRAIN / float($TRAIN)"
        python -c "print $OLDTTEST / float($TEST)"
        python -c "print ($OLDTTRAIN + $OLDTTEST) / float($TRAIN + $TEST)"
        python confusion.py
        echo '-'
        #diff -s OUTPUT.txt OUTPUTOLD.txt
        let WORKER_COUNTER=WORKER_COUNTER+1
    done
    WORKER_COUNTER=0
    let INPUT_COUNTER=INPUT_COUNTER+1
done

rm pendigitstrain?.arff
rm yeasttrain?.arff
rm dextertrain?.arff
rm amazontrain?.arff

