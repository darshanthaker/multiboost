/*
 *
 *    MultiBoost - Multi-purpose boosting package
 *
 *    Copyright (C)        AppStat group
 *                         Laboratoire de l'Accelerateur Lineaire
 *                         Universite Paris-Sud, 11, CNRS
 *
 *    This file is part of the MultiBoost library
 *
 *    This library is free software; you can redistribute it 
 *    and/or modify it under the terms of the GNU General Public
 *    License as published by the Free Software Foundation
 *    version 2.1 of the License.
 *
 *    This library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public
 *    License along with this library; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin St, 5th Floor, Boston, MA 02110-1301 USA
 *
 *    Contact: : multiboost@googlegroups.com
 *
 *    For more information and up-to-date version, please visit
 *        
 *                       http://www.multiboost.org/
 *
 */
#include "WeakLearners/BaseLearner.h"
#include "IO/InputData.h"
#include "Utils/Utils.h"
#include "IO/Serialization.h"
#include "IO/OutputInfo.h"
#include "Classifiers/AdaBoostPLClassifier.h"
#include "StrongLearners/AdaBoostPLLearner.h"
#include "Classifiers/ExampleResults.h"

#include "WeakLearners/SingleStumpLearner.h" // for saveSingleStumpFeatureData
#include "Timer.h"

#include <iomanip> // for setw
#include <cmath> // for setw
#include <functional>
#include <vector>
#include <iostream>
#include <string>
#include<pthread.h>

namespace MultiBoost {


    pthread_barrier_t barrier;
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------

    AdaBoostPLClassifier::AdaBoostPLClassifier(const nor_utils::Args &args, int verbose)
        : _verbose(verbose), _args(args)
    {
        // The file with the step-by-step information
        if ( args.hasArgument("outputinfo") )
            args.getValue("outputinfo", 0, _outputInfoFile);
        else
            _outputInfoFile = OUTPUT_NAME;
        if (args.hasArgument("nworkers2"))
            args.getValue("nworkers2", 0, _nThreads);
    
        if (args.hasArgument("nworkers"))
            args.getValue("nworkers", 0, _nWorkers);

    }

    // -------------------------------------------------------------------------

    void AdaBoostPLClassifier::run(const string& dataFileName, const string& shypFileName,
            int numIterations, const string& outResFileName, int numRanksEnclosed)
    {
        InputData* pData = loadInputData(dataFileName, shypFileName);

        if (_verbose > 0)
            cout << "Loading strong hypothesis..." << flush;

        // The class that loads the weak hypotheses
        UnSerialization us;

        // Where to put the weak hypotheses
        vector<BaseLearner*> weakHypotheses;

        // loads them
        us.loadHypotheses(shypFileName, weakHypotheses, pData);

        // where the results go
        vector< ExampleResults* > results;

        if (_verbose > 0)
            cout << "Classifying..." << flush;

        // get the results
        computeResults( pData, weakHypotheses, results, numIterations );

        const int numClasses = pData->getNumClasses();

        if (_verbose > 0)
        {            

            // well.. if verbose = 0 no results are displayed! :)
            cout << "Done!" << endl;

            vector< vector<float> > rankedError(numRanksEnclosed);

            // Get the per-class error for the numRanksEnclosed-th ranks
            for (int i = 0; i < numRanksEnclosed; ++i)
                getClassError( pData, results, rankedError[i], i );

            // output it
            cout << endl;
            cout << "True Positive Rates" << endl;
            cout << "===================" << endl;

            for ( int l = 0; l < numClasses; ++l )
            {
                // first rank (winner): rankedError[0]
                cout << "Class '" << pData->getClassMap().getNameFromIdx(l) << "': "
                    << setprecision(4) << (1 - rankedError[0][l]) * 100 << "%";

                // output the others on its side
                if (numRanksEnclosed > 1 && _verbose > 1)
                {
                    cout << " (";
                    for (int i = 1; i < numRanksEnclosed; ++i)
                        cout << " " << i+1 << ":[" << setprecision(4) << (1 - rankedError[i][l]) * 100 << "%]";
                    cout << " )";
                }

                cout << endl;
            }

            // the overall error
            cout << "\n--> Multi-Class Error Rate: "
                << setprecision(4) << getOverallError(pData, results, 0) * 100 << "%";

            // output the others on its side
            if (numRanksEnclosed > 1 && _verbose > 1)
            {
                cout << " (";
                for (int i = 1; i < numRanksEnclosed; ++i)
                    cout << " " << i+1 << ":[" << setprecision(4) << getOverallError(pData, results, i) * 100 << "%]";
                cout << " )";
            }

            cout << endl;

        } // verbose


        // If asked output the results
        if ( !outResFileName.empty() )
        {
            const int numExamples = pData->getNumExamples();
            ofstream outRes(outResFileName.c_str());

            outRes << "Instance" << '\t' << "Forecast" << '\t' << "Labels" << '\n';

            string exampleName;

            for (int i = 0; i < numExamples; ++i)
            {
                // output the name if it exists, otherwise the number
                // of the example
                exampleName = pData->getExampleName(i);
                if ( exampleName.empty() )
                    outRes << i << '\t';
                else
                    outRes << exampleName << '\t';

                // output the predicted class
                outRes << pData->getClassMap().getNameFromIdx( results[i]->getWinner().first ) << '\t';

                outRes << '|';

                vector<Label>& labels = pData->getLabels(i);
                for (vector<Label>::iterator lIt=labels.begin(); lIt != labels.end(); ++lIt) {
                    if (lIt->y>0)
                    {
                        outRes << ' ' << pData->getClassMap().getNameFromIdx(lIt->idx);
                    }
                }

                outRes << endl;
            }

            if (_verbose > 0)
                cout << "\nPredictions written on file <" << outResFileName << ">!" << endl;

        }


        // delete the input data file
        if (pData)
            delete pData;

        vector<ExampleResults*>::iterator it;
        for (it = results.begin(); it != results.end(); ++it)
            delete (*it);
    }

    // -------------------------------------------------------------------------

    void AdaBoostPLClassifier::printConfusionMatrix(const string& dataFileName, const string& shypFileName)
    {
        InputData* pData = loadInputData(dataFileName, shypFileName);

        if (_verbose > 0)
            cout << "Loading strong hypothesis..." << flush;

        // The class that loads the weak hypotheses
        UnSerialization us;

        // Where to put the weak hypotheses
        vector<BaseLearner*> weakHypotheses;

        // loads them
        us.loadHypotheses(shypFileName, weakHypotheses, pData);

        // where the results go
        vector< ExampleResults* > results;

        if (_verbose > 0)
            cout << "Classifying..." << flush;

        // get the results
        computeResults( pData, weakHypotheses, results, (int)weakHypotheses.size());


        const int numClasses = pData->getNumClasses();
        const int numExamples = pData->getNumExamples();

        if (_verbose > 0)
            cout << "Done!" << endl;

        const int colSize = 7;

        if (_verbose > 0)
        {
            cout << "Raw Confusion Matrix:\n";
            cout << setw(colSize) << "Prediction       ";

            for (int l = 0; l < numClasses; ++l)
                cout << setw(colSize) << nor_utils::getAlphanumeric(l);

            cout << "\nTruth\n";

            for (int l = 0; l < numClasses; ++l)
            {
                vector<int> winnerCount(numClasses, 0);
                for (int i = 0; i < numExamples; ++i)
                {
                    if ( pData->hasPositiveLabel(i, l) )
                        ++winnerCount[ results[i]->getWinner().first ];
                }

                // class
                cout << setw(colSize) << "                " << nor_utils::getAlphanumeric(l);

                for (int j = 0; j < numClasses; ++j)
                    cout << setw(colSize) << winnerCount[j];

                cout << endl;
            }

        }

        cout << "\nMatrix Key:\n";

        // Print the legend
        for (int l = 0; l < numClasses; ++l)
            cout << setw(5) << nor_utils::getAlphanumeric(l) << ": " <<
                pData->getClassMap().getNameFromIdx(l) << "\n";

        // delete the input data file
        if (pData)
            delete pData;

        vector<ExampleResults*>::iterator it;
        for (it = results.begin(); it != results.end(); ++it)
            delete (*it);
    }

    void* startMerge(void* arg)  
    {
        ThreadInfoPL *info  = (ThreadInfoPL*) arg;
        int newtid = info->tid;
        int newnumIterations = info->numIterations;
        int newlow = info->low;
        int newhigh = info->high;
        int newnumClasses = info->numClasses;
        int newnumExamples = info->numExamples;
        int newnumWorkers = info->numWorkers;
        vector<ExampleResults*>& newresults = info->results;
        vector<WeakOutput>& newweakOutputs = info->weakOutputs;
        InputData *newpData = info->pData;
        // for every point

        for (int i = info->low; i < info->high; ++i)
        {
            // a reference for clarity and speed
            vector<AlphaReal>& currVotesVector = (info->results)[i]->getVotesVector();

            // for every point
            for (int t = 0; t < info->numIterations; ++t)
            {
                // for every class
                for (int l = 0; l < info->numClasses; ++l)
                {
                    vector<BaseLearner*> column;
                    AlphaReal alpha  = 0;
                    for (int m = 0; m < info->numWorkers; ++m)
                    {
                        //change to use WeakOutputs
                        WeakOutput weakHyp = (info->weakOutputs)[m];
                        column.push_back(weakHyp.weakHypotheses[t]);
                        alpha += weakHyp.weakHypotheses[t]->getAlpha();
                    }
                    alpha /= (AlphaReal) info->numWorkers;
                    //printf("alpha = %f\n", alpha);

                    int mergeVal;

                    int counter = 0;
                    vector<BaseLearner*>::const_iterator whyIt;

                    int m;
                    for (whyIt = column.begin(), m = 0; whyIt != column.end() && m < info->numWorkers; ++whyIt, ++m) {
                        BaseLearner* currWeakHypothesis = *whyIt;
                        if (currWeakHypothesis->classify(info->pData, i, l) > 0) counter++;
                        else counter--;
                    }
                    if (counter > 0) mergeVal =  1;
                    else if (counter == 0) mergeVal = 0;
                    else mergeVal = -1;

                    //printf("mergeval = %d\n", mergeVal);
                    currVotesVector[l] += alpha * mergeVal;
                    //printf("i = %d, l = %d\n", i, l);

                    // currVotesVector[l] += alpha * merge(info->pData, column, i, l, info->numWorkers);
                }
            }
        }
        pthread_barrier_wait(&barrier);
    }

    //Takes in a array of weakHypotheses rather than just one vector of weak Hypotheses
    void AdaBoostPLClassifier::computeMergeResults(InputData *pData, vector<WeakOutput>& weakOutputs, vector< ExampleResults* >& results, int numIterations, int numWorkers)
    {
        printf("COMPUTING MERGE RESULTS\n");
        //assert( !weakHypotheses.empty() );

        const int numClasses = pData->getNumClasses();
        const int numExamples = pData->getNumExamples();

        printf("No Segfault...yet\n");	

        // Initialize the output info
        OutputInfo* pOutInfo = NULL;

        if ( !_outputInfoFile.empty() )
        {
            pOutInfo = new OutputInfo(_args);
        }


        // Creating the results structures. See file Structures.h for the
        // PointResults structure
        results.clear();
        results.reserve(numExamples);
        for (int i = 0; i < numExamples; ++i)
            results.push_back( new ExampleResults(i, numClasses) );

        printf("No Segfault...yet again\n");	

        // iterator over all the weak hypotheses
        vector<BaseLearner*>::const_iterator whyIt;

        if ( pOutInfo )
        {
            pOutInfo->initialize( pData );
            pOutInfo->outputHeader(pData->getClassMap(),
                    true, // output iterations
                    false, // output time
                    true // endline
                    );
        }

        pthread_t threads[numWorkers];
        pthread_barrier_init(&barrier, NULL, _nThreads + 1);
        ThreadInfoPL *thread_infos[_nThreads];
        int rc;
        printf("NumExamples is: %d\n", numExamples);
        for (int i = 0; i < _nThreads; i++)
        {
            int low = (i * numExamples/_nThreads);
            int high;
            if (i == _nThreads - 1)
            {
                high = numExamples;
            }
            else
            {
                high = ((i+1) * numExamples/_nThreads);
            }
            printf("Low = %d, High = %d\n", low, high);
            thread_infos[i] = new ThreadInfoPL(i, numClasses, numIterations,
                    numWorkers, numExamples, results, weakOutputs, pData, low, high);
            //printf("Info low = %d, Info high = %d\n", thread_infos[]->low, thread_infos->high);
            rc = pthread_create(&threads[i], NULL, startMerge, (void *)(thread_infos[i])); 
            if (rc) {
                printf("Error creating thread\n");
                exit(-1);
            }
        }

        pthread_barrier_wait(&barrier);


    }

    int AdaBoostPLClassifier::merge(InputData *pData, vector<BaseLearner*>& column, int point, int label, int numWorkers) {
        int counter = 0;
        vector<BaseLearner*>::const_iterator whyIt;

        //for (whyIt = weakHypotheses.begin(), t = 0;
        //     whyIt != weakHypotheses.end() && t < numIterations; ++whyIt, ++t)
        int m;
        for (whyIt = column.begin(), m = 0; whyIt != column.end() && m < numWorkers; ++whyIt, ++m) {
            BaseLearner* currWeakHypothesis = *whyIt;
            //currVotesVector[l] += alpha * currWeakHyp->classify(pData, i, l);
            //TODO: Find out where the classify method is defined...
            if (currWeakHypothesis->classify(pData, point, label) > 0) counter++;
            else counter--;
        }
        if (counter > 0) return 1;
        if (counter == 0) return 0;
        return -1;
    }


    // -------------------------------------------------------------------------

    void AdaBoostPLClassifier::saveConfusionMatrix(const string& dataFileName, const string& shypFileName,
            const string& outFileName)
    {
        InputData* pData = loadInputData(dataFileName, shypFileName);
        ggc::Timer t("testing");

        if (_verbose > 0)
            cout << "Loading strong hypothesis..." << flush;

        vector<WeakOutput> weakOutputs;

        for (int m = 0; m < _nWorkers; m++)
        {
            // The class that loads the weak hypotheses
            UnSerialization us;

            // Where to put the weak hypotheses
            vector<BaseLearner*> weakHypotheses;

            WeakOutput weakOutput;

            string result; 
            ostringstream convert;   // stream used for the conversion
            convert << m;
            result = convert.str();
            string newName = shypFileName.substr(0, shypFileName.find(".xml") - 1);
            string newShypfileName = newName + result + ".xml";

            printf("shypFilename = %s\n", newShypfileName.c_str());

            // loads them
            us.loadHypotheses(newShypfileName, weakHypotheses, pData);

            weakOutput.weakHypotheses = weakHypotheses;		

            weakOutputs.push_back(weakOutput);
        }
        // where the results go
        vector< ExampleResults* > results;

        if (_verbose > 0)
            cout << "Classifying..." << flush;


        //pdata, weakoutputs, results, numiterations, numworkers
        t.start();
        computeMergeResults( pData, weakOutputs, results, (int) weakOutputs[0].weakHypotheses.size(), (int) weakOutputs.size());
        t.stop();
        printf("Testing time is: %llu \n", t.duration());
        // get the results
        //computeResults( pData, weakHypotheses, results, (int)weakHypotheses.size() );

        const int numClasses = pData->getNumClasses();
        const int numExamples = pData->getNumExamples();

        ofstream outFile(outFileName.c_str());

        //////////////////////////////////////////////////////////////////////////

        for (int l = 0; l < numClasses; ++l)
            outFile << '\t' << pData->getClassMap().getNameFromIdx(l);
        outFile << endl;

        for (int l = 0; l < numClasses; ++l)
        {
            vector<int> winnerCount(numClasses, 0);
            for (int i = 0; i < numExamples; ++i)
            {
                if ( pData->hasPositiveLabel(i,l) )
                    ++winnerCount[ results[i]->getWinner().first ];
            }

            // class name
            outFile << pData->getClassMap().getNameFromIdx(l);

            for (int j = 0; j < numClasses; ++j)
                outFile << '\t' << winnerCount[j];

            outFile << endl;
        }

        //////////////////////////////////////////////////////////////////////////

        if (_verbose > 0)
            cout << "Done!" << endl;



        // delete the input data file
        if (pData)
            delete pData;

        vector<ExampleResults*>::iterator it;
        for (it = results.begin(); it != results.end(); ++it)
        {
            delete (*it);
        }
    }

    // -------------------------------------------------------------------------

    void AdaBoostPLClassifier::savePosteriors(const string& dataFileName, const string& shypFileName,
            const string& outFileName, int numIterations, int period)
    {
        InputData* pData = loadInputData(dataFileName, shypFileName);

        if (_verbose > 0)
            cout << "Loading strong hypothesis..." << flush;

        // The class that loads the weak hypotheses
        UnSerialization us;

        // Where to put the weak hypotheses
        vector<BaseLearner*> weakHypotheses;

        // loads them
        us.loadHypotheses(shypFileName, weakHypotheses, pData);

        // where the results go
        vector< ExampleResults* > results;

        if (_verbose > 0)
            cout << "Classifying..." << flush;

        if ( period == 0 )
            period=numIterations;

        // get the results
        computeResults( pData, weakHypotheses, results, period );

        const int numClasses = pData->getNumClasses();
        const int numExamples = pData->getNumExamples();

        ofstream outFile(outFileName.c_str());
        string exampleName;

        if (_verbose > 0)
            cout << "Output posteriors..." << flush;

        if (period<numIterations)
            outFile << period << endl;

        for (int i = 0; i < numExamples; ++i)
        {
            // output the name if it exists, otherwise the number
            // of the example
            exampleName = pData->getExampleName(i);
            if ( !exampleName.empty() )
                outFile << exampleName << ',';

            // output the posteriors
            outFile << results[i]->getVotesVector()[0];
            for (int l = 1; l < numClasses; ++l)
                outFile << ',' << results[i]->getVotesVector()[l];
            outFile << '\n';
        }


        for (int p=period; p<numIterations; p+=period )
        {
            if ( (p+period) > weakHypotheses.size() ) break;

            continueComputingResults(pData, weakHypotheses, results, p, p+period );
            if ( _verbose > 0) {
                cout << "Write out the posterios for iteration " << p << endl;
            }
            outFile << p+period << endl;
            for (int i = 0; i < numExamples; ++i)
            {
                // output the name if it exists, otherwise the number
                // of the example
                exampleName = pData->getExampleName(i);
                if ( !exampleName.empty() )
                    outFile << exampleName << ',';

                // output the posteriors
                outFile << results[i]->getVotesVector()[0];
                for (int l = 1; l < numClasses; ++l)
                    outFile << ',' << results[i]->getVotesVector()[l];
                outFile << '\n';
            }

        }

        if (_verbose > 0)
            cout << "Done!" << endl;

        if (_verbose > 1)
        {
            cout << "\nClass order (You can change it in the header of the data file):" << endl;
            for (int l = 0; l < numClasses; ++l)
                cout << "- " << pData->getClassMap().getNameFromIdx(l) << endl;
        }

        // delete the input data file
        if (pData)
            delete pData;

        vector<ExampleResults*>::iterator it;
        for (it = results.begin(); it != results.end(); ++it)
            delete (*it);
    }


    // -------------------------------------------------------------------------


    void AdaBoostPLClassifier::saveCalibratedPosteriors(const string& dataFileName, const string& shypFileName,
            const string& outFileName, int numIterations)
    {
        InputData* pData = loadInputData(dataFileName, shypFileName);

        if (_verbose > 0)
            cout << "Loading strong hypothesis..." << flush;

        // The class that loads the weak hypotheses
        UnSerialization us;

        // Where to put the weak hypotheses
        vector<BaseLearner*> weakHypotheses;

        // loads them
        us.loadHypotheses(shypFileName, weakHypotheses, pData);

        // where the results go
        vector< ExampleResults* > results;

        if (_verbose > 0)
            cout << "Classifying..." << flush;

        // get the results
        computeResults( pData, weakHypotheses, results, numIterations );

        const int numClasses = pData->getNumClasses();
        const int numExamples = pData->getNumExamples();

        ofstream outFile(outFileName.c_str());
        string exampleName;

        if (_verbose > 0)
            cout << "Output posteriors..." << flush;

        for (int i = 0; i < numExamples; ++i)
        {
            // output the name if it exists, otherwise the number
            // of the example
            exampleName = pData->getExampleName(i);
            if ( !exampleName.empty() )
                outFile << exampleName << ',';

            // output the posteriors
            outFile << results[i]->getVotesVector()[0];
            for (int l = 1; l < numClasses; ++l)
                outFile << ',' << results[i]->getVotesVector()[l];
            outFile << '\n';
        }

        if (_verbose > 0)
            cout << "Done!" << endl;

        if (_verbose > 1)
        {
            cout << "\nClass order (You can change it in the header of the data file):" << endl;
            for (int l = 0; l < numClasses; ++l)
                cout << "- " << pData->getClassMap().getNameFromIdx(l) << endl;
        }

        // delete the input data file
        if (pData)
            delete pData;

        vector<ExampleResults*>::iterator it;
        for (it = results.begin(); it != results.end(); ++it)
            delete (*it);
    }



    // -------------------------------------------------------------------------

    void AdaBoostPLClassifier::saveLikelihoods(const string& dataFileName, const string& shypFileName,
            const string& outFileName, int numIterations)
    {
        InputData* pData = loadInputData(dataFileName, shypFileName);

        if (_verbose > 0)
            cout << "Loading strong hypothesis..." << flush;

        // The class that loads the weak hypotheses
        UnSerialization us;

        // Where to put the weak hypotheses
        vector<BaseLearner*> weakHypotheses;

        // loads them
        us.loadHypotheses(shypFileName, weakHypotheses, pData);

        // where the results go
        vector< ExampleResults* > results;

        if (_verbose > 0)
            cout << "Classifying..." << flush;

        const int numClasses = pData->getNumClasses();
        const int numExamples = pData->getNumExamples();


        ofstream outFile(outFileName.c_str());
        string exampleName;

        if (_verbose > 0)
            cout << "Output likelihoods..." << flush;

        // get the results
        /////////////////////////////////////////////////////////////////////
        // computeResults( pData, weakHypotheses, results, numIterations );
        assert( !weakHypotheses.empty() );

        // Initialize the output info
        OutputInfo* pOutInfo = NULL;

        if ( !_outputInfoFile.empty() )
            pOutInfo = new OutputInfo(_args);

        // Creating the results structures. See file Structures.h for the
        // PointResults structure
        results.clear();
        results.reserve(numExamples);
        for (int i = 0; i < numExamples; ++i)
            results.push_back( new ExampleResults(i, numClasses) );

        // sum votes for classes
        vector< AlphaReal > votesForExamples( numClasses );
        vector< AlphaReal > expVotesForExamples( numClasses );

        // iterator over all the weak hypotheses
        vector<BaseLearner*>::const_iterator whyIt;
        int t;

        pOutInfo->initialize( pData );

        // for every feature: 1..T
        for (whyIt = weakHypotheses.begin(), t = 0;
                whyIt != weakHypotheses.end() && t < numIterations; ++whyIt, ++t)
        {
            BaseLearner* currWeakHyp = *whyIt;
            AlphaReal alpha = currWeakHyp->getAlpha();

            // for every point
            for (int i = 0; i < numExamples; ++i)
            {
                // a reference for clarity and speed
                vector<AlphaReal>& currVotesVector = results[i]->getVotesVector();

                // for every class
                for (int l = 0; l < numClasses; ++l)
                    currVotesVector[l] += alpha * currWeakHyp->classify(pData, i, l);
            }

            // if needed output the step-by-step information
            if ( pOutInfo )
            {
                pOutInfo->outputIteration(t);
                pOutInfo->outputCustom(pData, currWeakHyp);

                // Margins and edge requires an update of the weight,
                // therefore I keep them out for the moment
                //outInfo.outputMargins(pData, currWeakHyp);
                //outInfo.outputEdge(pData, currWeakHyp);

                pOutInfo->endLine();

            } // for (int i = 0; i < numExamples; ++i)
            // calculate likelihoods from votes

            fill( votesForExamples.begin(), votesForExamples.end(), 0.0 );
            AlphaReal lLambda = 0.0;
            for (int i = 0; i < numExamples; ++i)
            {
                // a reference for clarity and speed
                vector<AlphaReal>& currVotesVector = results[i]->getVotesVector();
                AlphaReal sumExp = 0.0;
                // for every class
                for (int l = 0; l < numClasses; ++l)
                {
                    expVotesForExamples[l] =  exp( currVotesVector[l] ) ;
                    sumExp += expVotesForExamples[l];
                }

                if ( sumExp > numeric_limits<AlphaReal>::epsilon() )
                {
                    for (int l = 0; l < numClasses; ++l)
                    {
                        expVotesForExamples[l] /= sumExp;
                    }
                }

                Example ex = pData->getExample( results[i]->getIdx() );
                vector<Label> labs = ex.getLabels();
                AlphaReal m = numeric_limits<AlphaReal>::infinity();
                for (int l = 0; l < numClasses; ++l)
                {
                    if ( labs[l].y > 0 )
                    {
                        if ( expVotesForExamples[l] > numeric_limits<AlphaReal>::epsilon() )
                        {
                            AlphaReal logVal = log( expVotesForExamples[l] );

                            if ( logVal != m ) {
                                lLambda += ( ( 1.0/(AlphaReal)numExamples ) * logVal );
                            }
                        }
                    }
                }


            }


            outFile << t << "\t" << lLambda ;
            outFile << '\n';

            outFile.flush();
        }

        if (pOutInfo)
            delete pOutInfo;

        // computeResults( pData, weakHypotheses, results, numIterations );
        ///////////////////////////////////////////////////////////////////////////////////


        /*
           for (int i = 0; i < numExamples; ++i)
           {
        // output the name if it exists, otherwise the number
        // of the example
        exampleName = pData->getExampleName(i);
        if ( !exampleName.empty() )
        outFile << exampleName << ',';

        // output the posteriors
        outFile << results[i]->getVotesVector()[0];
        for (int l = 1; l < numClasses; ++l)
        outFile << ',' << results[i]->getVotesVector()[l];
        outFile << '\n';
        }
        */

        if (_verbose > 0)
            cout << "Done!" << endl;

        if (_verbose > 1)
        {
            cout << "\nClass order (You can change it in the header of the data file):" << endl;
            for (int l = 0; l < numClasses; ++l)
                cout << "- " << pData->getClassMap().getNameFromIdx(l) << endl;
        }

        // delete the input data file
        if (pData)
            delete pData;

        vector<ExampleResults*>::iterator it;
        for (it = results.begin(); it != results.end(); ++it)
            delete (*it);
    }


    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------

    InputData* AdaBoostPLClassifier::loadInputData(const string& dataFileName, const string& shypFileName)
    {
        // open file
        ifstream inFile(shypFileName.c_str());
        if (!inFile.is_open())
        {
            cerr << "ERROR: Cannot open strong hypothesis file <" << shypFileName << ">!" << endl;
            exit(1);
        }

        // Declares the stream tokenizer
        nor_utils::StreamTokenizer st(inFile, "<>\n\r\t");

        // Move until it finds the multiboost tag
        if ( !UnSerialization::seekSimpleTag(st, "multiboost") )
        {
            // no multiboost tag found: this is not the correct file!
            cerr << "ERROR: Not a valid MultiBoost Strong Hypothesis file!!" << endl;
            exit(1);
        }

        // Move until it finds the algo tag
        string basicLearnerName = UnSerialization::seekAndParseEnclosedValue<string>(st, "algo");

        // Check if the weak learner exists
        if ( !BaseLearner::RegisteredLearners().hasLearner(basicLearnerName) )
        {
            cerr << "ERROR: Weak learner <" << basicLearnerName << "> not registered!!" << endl;
            exit(1);
        }

        // get the training input data, and load it
        BaseLearner* baseLearner = BaseLearner::RegisteredLearners().getLearner(basicLearnerName);
        baseLearner->initLearningOptions(_args);
        InputData* pData = baseLearner->createInputData();

        // set the non-default arguments of the input data
        pData->initOptions(_args);
        // load the data
        pData->load(dataFileName, IT_TEST, _verbose);

        return pData;
    }

    // -------------------------------------------------------------------------

    // Returns the results into ptRes
    void AdaBoostPLClassifier::computeResults(InputData* pData, vector<BaseLearner*>& weakHypotheses,
            vector< ExampleResults* >& results, int numIterations)
    {
        assert( !weakHypotheses.empty() );

        const int numClasses = pData->getNumClasses();
        const int numExamples = pData->getNumExamples();

        // Initialize the output info
        OutputInfo* pOutInfo = NULL;

        if ( !_outputInfoFile.empty() )
        {
            pOutInfo = new OutputInfo(_args);

        }


        // Creating the results structures. See file Structures.h for the
        // PointResults structure
        results.clear();
        results.reserve(numExamples);
        for (int i = 0; i < numExamples; ++i)
            results.push_back( new ExampleResults(i, numClasses) );

        // iterator over all the weak hypotheses
        vector<BaseLearner*>::const_iterator whyIt;
        int t;

        if ( pOutInfo )
        {
            pOutInfo->initialize( pData );
            pOutInfo->outputHeader(pData->getClassMap(),
                    true, // output iterations
                    false, // output time
                    true // endline
                    );
        }

        // for every feature: 1..T
        for (whyIt = weakHypotheses.begin(), t = 0;
                whyIt != weakHypotheses.end() && t < numIterations; ++whyIt, ++t)
        {
            BaseLearner* currWeakHyp = *whyIt;
            AlphaReal alpha = currWeakHyp->getAlpha();

            // for every point
            for (int i = 0; i < numExamples; ++i)
            {
                // a reference for clarity and speed
                vector<AlphaReal>& currVotesVector = results[i]->getVotesVector();

                // for every class
                for (int l = 0; l < numClasses; ++l)
                    currVotesVector[l] += alpha * currWeakHyp->classify(pData, i, l);
            }

            // if needed output the step-by-step information
            if ( pOutInfo )
            {
                pOutInfo->outputIteration(t);
                //                              pOutInfo->outputError(pData, currWeakHyp);
                //                              pOutInfo->outTPRFPR(pData);
                //pOutInfo->outputBalancedError(pData, currWeakHyp);
                //                              if ( ( t % 1 ) == 0 ) {
                //                                      pOutInfo->outputROC(pData);
                //                              }

                pOutInfo->outputCustom(pData, currWeakHyp);
                // Margins and edge requires an update of the weight,
                // therefore I keep them out for the moment
                //outInfo.outputMargins(pData, currWeakHyp);
                //outInfo.outputEdge(pData, currWeakHyp);
                pOutInfo->endLine();
            }
        }

        if (pOutInfo)
            delete pOutInfo;

    }

    // -------------------------------------------------------------------------

    // Continue returns the results into ptRes for savePosteriors
    // must be called the computeResult first!!!
    void AdaBoostPLClassifier::continueComputingResults(InputData* pData, vector<BaseLearner*>& weakHypotheses,
            vector< ExampleResults* >& results, int fromIteration, int toIteration)
    {
        assert( !weakHypotheses.empty() );

        const int numClasses = pData->getNumClasses();
        const int numExamples = pData->getNumExamples();


        // iterator over all the weak hypotheses
        vector<BaseLearner*>::const_iterator whyIt;
        int t;

        for (whyIt = weakHypotheses.begin(), t = 0;
                whyIt != weakHypotheses.end() && t < fromIteration; ++whyIt, ++t) {}

        // for every feature: 1..T
        for (;whyIt != weakHypotheses.end() && t < toIteration; ++whyIt, ++t)
        {
            BaseLearner* currWeakHyp = *whyIt;
            AlphaReal alpha = currWeakHyp->getAlpha();

            // for every point
            for (int i = 0; i < numExamples; ++i)
            {
                // a reference for clarity and speed
                vector<AlphaReal>& currVotesVector = results[i]->getVotesVector();

                // for every class
                for (int l = 0; l < numClasses; ++l)
                    currVotesVector[l] += alpha * currWeakHyp->classify(pData, i, l);
            }
        }

    }

    // -------------------------------------------------------------------------

    float AdaBoostPLClassifier::getOverallError( InputData* pData, const vector<ExampleResults*>& results,
            int atLeastRank )
    {
        const int numExamples = pData->getNumExamples();
        int numErrors = 0;

        assert(atLeastRank >= 0);

        for (int i = 0; i < numExamples; ++i)
        {
            // if the actual class is not the one with the highest vote in the
            // vote vector, then it is an error!
            if ( !results[i]->isWinner( pData->getExample(i), atLeastRank ) )
                ++numErrors;
        }

        // makes the error between 0 and 1
        return (float)numErrors / (float)numExamples;
    }

    // -------------------------------------------------------------------------

    void AdaBoostPLClassifier::getClassError( InputData* pData, const vector<ExampleResults*>& results,
            vector<float>& classError, int atLeastRank )
    {
        const int numExamples = pData->getNumExamples();
        const int numClasses = pData->getNumClasses();

        classError.resize( numClasses, 0 );

        assert(atLeastRank >= 0);

        for (int i = 0; i < numExamples; ++i)
        {
            // if the actual class is not the one with the highest vote in the
            // vote vector, then it is an error!
            if ( !results[i]->isWinner( pData->getExample(i), atLeastRank ) )
            {
                const vector<Label>& labels = pData->getLabels(i);
                vector<Label>::const_iterator lIt;
                for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
                {
                    if ( lIt->y > 0 )
                        ++classError[ lIt->idx ];
                }
                //++classError[ pData->getClass(i) ];
            }
        }

        // makes the error between 0 and 1
        for (int l = 0; l < numClasses; ++l)
            classError[l] /= (float)pData->getNumExamplesPerClass(l);
    }

    // -------------------------------------------------------------------------

} // end of namespace MultiBoost
