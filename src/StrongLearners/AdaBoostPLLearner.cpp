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


#include <ctime> // for time
#include <cmath> // for exp
#include <fstream> // for ofstream of the step-by-step data
#include <iostream>
#include <limits>
#include <iomanip> // setprecision

#include "Utils/Utils.h" // for addAndCheckExtension
#include "Defaults.h" // for defaultLearner
#include "IO/OutputInfo.h"
#include "IO/InputData.h"
#include "IO/Serialization.h" // to save the found strong hypothesis

#include "WeakLearners/BaseLearner.h"
#include "StrongLearners/AdaBoostPLLearner.h"
#include "StrongLearners/AdaBoostMHLearner.h"

#include "Classifiers/AdaBoostPLClassifier.h"
#include "Timer.h"

pthread_barrier_t workerBarrier; 
namespace MultiBoost {

    PartitionsData *partition_data;
    WeakOutput **weakOutputs;
    // -----------------------------------------------------------------------------------


    void AdaBoostPLLearner::getArgs(const nor_utils::Args& args)
    {

        ///////////////////////////////////////////////////
        // Number of workers
        if ( args.hasArgument("nworkers") )
            args.getValue("nworkers", 0, _nWorkers);

        if ( args.hasArgument("verbose") )
            args.getValue("verbose", 0, _verbose);

        // The file with the step-by-step information

        if ( args.hasArgument("outputinfo") )
            args.getValue("outputinfo", 0, _outputInfoFile);
        else
            _outputInfoFile = OUTPUT_NAME;

        ///////////////////////////////////////////////////
        // get the output strong hypothesis file name, if given
        if ( args.hasArgument("shypname") )
            args.getValue("shypname", 0, _shypFileName);
        else
            _shypFileName = string(SHYP_NAME);

        _shypFileName = nor_utils::addAndCheckExtension(_shypFileName, SHYP_EXTENSION);

        ///////////////////////////////////////////////////
        // get the output strong hypothesis file name, if given
        if ( args.hasArgument("shypcomp") )
            args.getValue("shypcomp", 0, _isShypCompressed );
        else
            _isShypCompressed = false;


        ///////////////////////////////////////////////////
        // Set time limit
        if ( args.hasArgument("timelimit") )
        {
            args.getValue("timelimit", 0, _maxTime);   
            if (_verbose > 1)    
                cout << "--> Overall Time Limit: " << _maxTime << " minutes" << endl;
        }

        // Set the value of theta
        if ( args.hasArgument("edgeoffset") )
            args.getValue("edgeoffset", 0, _theta);  

        // Set the filename of the strong hypothesis file in the case resume is
        // called
        if ( args.hasArgument("resume") )
            args.getValue("resume", 0, _resumeShypFileName);

        // get the name of the learner
        _baseLearnerName = defaultLearner;
        if ( args.hasArgument("learnertype") )
            args.getValue("learnertype", 0, _baseLearnerName);

        _earlyStopping = false;
        // -train <dataFile> <nInterations>
        if ( args.hasArgument("train") )
        {
            args.getValue("train", 0, _trainFileName);
            args.getValue("train", 1, _numIterations);
        }
        // -traintest <trainingDataFile> <testDataFile> <nInterations>
        else if ( args.hasArgument("traintest") ) 
        {
            args.getValue("traintest", 0, _trainFileName);
            args.getValue("traintest", 1, _testFileName);
            args.getValue("traintest", 2, _numIterations);

            // --earlystopping <minIterations> <smoothingWindowRate> <maxLookaheadRate>
            if ( args.hasArgument("earlystopping") )
            {
                _earlyStopping = true;
                args.getValue("earlystopping", 0, _earlyStoppingMinIterations);
                args.getValue("earlystopping", 1, _earlyStoppingSmoothingWindowRate);
                args.getValue("earlystopping", 2, _earlyStoppingMaxLookaheadRate); 
                if (args.hasArgument("earlystoppingoutputinfo"))
                    args.getValue("earlystoppingoutputinfo", 0, _earlyStoppingOutputColumn);
                else _earlyStoppingOutputColumn = "e01";
            }
        }


        // --constant: check constant learner in each iteration
        if ( args.hasArgument("constant") )
            _withConstantLearner = true;

        // it recalculates the whole test output file
        if ( args.hasArgument("slowresumeprocess") ) {
            _fastResumeProcess = false;
        }

        // --weights <filename>
        if ( args.hasArgument("weights") ) {
            args.getValue("weights", 0, _weightFile );
        }
    }

    // -----------------------------------------------------------------------------------
    void AdaBoostPLLearner::createPartitions(const nor_utils::Args& args) {
        // Partition the original input_data file into multiple files
        // Call createInputData, initoptions, and load on these files and store
        // the result in a global array of inputData pointers.
        // The global array should be malloced in this method  
        partition_data = new PartitionsData(_nWorkers); 
        BaseLearner* pWeakHypothesisSource = 
            BaseLearner::RegisteredLearners().getLearner(_baseLearnerName);
        pWeakHypothesisSource->initLearningOptions(args);

        for (int i = 0; i < _nWorkers; i++) {
            string result; 
            ostringstream convert;   // stream used for the conversion
            convert << i;
            result = convert.str();
            string newName = _trainFileName.substr(0, _trainFileName.find(".arff"));
            partition_data->fileNames[i] = newName + result + ".arff";
            partition_data->outfiles[i].open(partition_data->fileNames[i].c_str());
        }

        ifstream infile; 
        string data;
        infile.open(_trainFileName.c_str());
        while (data.compare("@DATA") != 0) {
            getline(infile, data);
            for (int i = 0; i < _nWorkers; i++) {
                partition_data->outfiles[i] << data << '\n'; 
            }
        }

        int counter = 0;
        while (!infile.eof()) {
            getline(infile, data);
            partition_data->outfiles[counter % _nWorkers] << data << '\n';
            counter++;
        }

        //Close all the file streams
        infile.close();
        for (int j = 0; j < _nWorkers; j++) {
            partition_data->outfiles[j].close();
        }

        for (int k = 0; k < _nWorkers; k++) {
            string fileName = partition_data->fileNames[k];
            partition_data->partitions[k] = pWeakHypothesisSource->createInputData();
            partition_data->partitions[k]->initOptions(args);
            partition_data->partitions[k]->load(fileName, IT_TRAIN, _verbose);
        }
    }
    // -----------------------------------------------------------------------------------

    // -----------------------------------------------------------------------------------
    void AdaBoostPLLearner::deletePartitions() {
    }
    // -----------------------------------------------------------------------------------

    // -----------------------------------------------------------------------------------

    bool sortLearner(BaseLearner* learner1, BaseLearner* learner2)
    {
	return (learner1->getAlpha() < learner2->getAlpha());
    }

    void* startWorker(void* arg)  
    {
        ThreadInfo *info = (ThreadInfo *) arg;
        int tid = info->tid;
        string shypFileName = info->shypFileName;
        printf("[startWorker] tid = %d\n", tid);
        int numIterations = info->numIterations;
        string trainFileName = partition_data->fileNames[tid];
        InputData* pTrainingData = partition_data->partitions[tid];
        string result; 
        ostringstream convert;   // stream used for the conversion
        convert << tid;
        result = convert.str();
        string newName = shypFileName.substr(0, shypFileName.find(".xml"));
        string newFileName = newName + result + ".xml";

        AdaBoostMHLearner *MHLearner = new AdaBoostMHLearner();
        MHLearner->setParallel();
        MHLearner->setShypFileName(newFileName);
        MHLearner->run(info->args, pTrainingData, "SingleStumpLearner",
                numIterations, weakOutputs[tid]->weakHypotheses);
        pthread_barrier_wait(&workerBarrier);
	std::sort(weakOutputs[tid]->weakHypotheses.begin() ,weakOutputs[tid]->weakHypotheses.end(), sortLearner);
        pthread_exit(NULL);	
    }

    void AdaBoostPLLearner::run(const nor_utils::Args& args)
    {
        ggc::Timer t("training");
        t.start();
        // load the arguments
        this->getArgs(args);

        weakOutputs = new WeakOutput*[_nWorkers];

        for (int i = 0; i < _nWorkers; i++) {
            weakOutputs[i] = new WeakOutput();
        }
        // get the registered weak learner (type from name)
        BaseLearner* pWeakHypothesisSource = 
            BaseLearner::RegisteredLearners().getLearner(_baseLearnerName);
        //GenericStrongLearner *pModel = pWeakHypothesisSource->createGenericStrongLearner( args );
        // initialize learning options; normally it's done in the strong loop
        // also, here we do it for Product learners, so input data can be created
        pWeakHypothesisSource->initLearningOptions(args);

        /*BaseLearner* pConstantWeakHypothesisSource = 
          BaseLearner::RegisteredLearners().getLearner("ConstantLearner"); */
        t.stop();
        createPartitions(args);
        t.start();

        pthread_t threads[_nWorkers];
        int tid;

        if (pthread_barrier_init(&workerBarrier, NULL, _nWorkers + 1) != 0) {
            printf("\nbarrier init failed\n");
            return;
        }

        for (tid = 0; tid < _nWorkers; tid++)
        {		
            ThreadInfo *threadInfo = new ThreadInfo(tid, args, _numIterations, _shypFileName);
            pthread_create(&threads[tid], NULL, startWorker, (void*) threadInfo);
        }	
        pthread_barrier_wait(&workerBarrier);

	

        deletePartitions(); 
        t.stop();
	    printf("Training time is : %llu \n",t.duration());
    }

    // -------------------------------------------------------------------------

    void AdaBoostPLLearner::classify(const nor_utils::Args& args)
    {
        AdaBoostPLClassifier classifier(args, _verbose);

        // -test <dataFile> <shypFile>
        string testFileName = args.getValue<string>("test", 0);
        string shypFileName = args.getValue<string>("test", 1);
        int numIterations = args.getValue<int>("test", 2);

        string outResFileName;
        if ( args.getNumValues("test") > 3 )
            args.getValue("test", 3, outResFileName);

        classifier.run(testFileName, shypFileName, numIterations, outResFileName);
    }

    // -------------------------------------------------------------------------

    void AdaBoostPLLearner::doConfusionMatrix(const nor_utils::Args& args)
    {
        AdaBoostPLClassifier classifier(args, _verbose);

        // -cmatrix <dataFile> <shypFile>
        if ( args.getNumValues("cmatrix") == 2 )
        {
            string testFileName = args.getValue<string>("cmatrix", 0);
            string shypFileName = args.getValue<string>("cmatrix", 1);

            classifier.printConfusionMatrix(testFileName, shypFileName);
        }
        // -cmatrix <dataFile> <shypFile> <outFile>
        else if ( args.getNumValues("cmatrix") == 3)
        {
            string testFileName = args.getValue<string>("cmatrix", 0);
            string shypFileName = args.getValue<string>("cmatrix", 1);
            string outResFileName = args.getValue<string>("cmatrix", 2);

            classifier.saveConfusionMatrix(testFileName, shypFileName, outResFileName);
        }
    }

    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------

    void AdaBoostPLLearner::doPosteriors(const nor_utils::Args& args)
    {
        if ( args.hasArgument("verbose") )
            args.getValue("verbose", 0, _verbose);

        AdaBoostPLClassifier classifier(args, _verbose);
        int numofargs = args.getNumValues( "posteriors" );
        // -posteriors <dataFile> <shypFile> <outFile> <numIters>
        string testFileName = args.getValue<string>("posteriors", 0);
        string shypFileName = args.getValue<string>("posteriors", 1);
        string outFileName = args.getValue<string>("posteriors", 2);
        int numIterations = args.getValue<int>("posteriors", 3);
        int period = 0;

        if ( numofargs == 5 )
            period = args.getValue<int>("posteriors", 4);

        classifier.savePosteriors(testFileName, shypFileName, outFileName, numIterations, period);
    }

    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    AlphaReal AdaBoostPLLearner::updateWeights(OutputInfo* pOutInfo, InputData* pData, vector<BaseLearner*>& pWeakHypothesiss){
        const int numExamples = pData->getNumExamples();
        const int numClasses = pData->getNumClasses();

        // _hy will contain the margins
        _hy.resize(numExamples);
        for ( int i = 0; i < numExamples; ++i){
            _hy[i].resize(numClasses);
            fill( _hy[i].begin(), _hy[i].end(), 0.0 );

            vector<Label>& labels = pData->getLabels(i);
            // initializing to log weights
            for (vector<Label>::iterator lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                lIt->weight = log(lIt->weight);
            }
        }


        if (_verbose > 0)
            cout << ": 0%." << flush;

        const int numIters = static_cast<int>(_foundHypotheses.size());
        const int step = numIters < 5 ? 1 : numIters / 5;

        vector<BaseLearner*>::iterator it;
        int t;
        for(t = 0, it = pWeakHypothesiss.begin(); it != pWeakHypothesiss.end(); it++, t++)
        {
            if (_verbose > 1)
            {
                if ((t + 1) % 1000 == 0)
                    cout << "." << flush;
                if ((t + 1) % step == 0)
                {
                    float progress = static_cast<float>(t) / static_cast<float>(numIters) * 100.0;                             
                    cout << "." << setprecision(2) << progress << "%." << flush;
                }
            }

            BaseLearner* pWeakHypothesis = *it;
            const AlphaReal alpha = pWeakHypothesis->getAlpha();
            AlphaReal hx;
            for (int i = 0; i < numExamples; ++i)
            {
                vector<Label>& labels = pData->getLabels(i);
                // accumulating margins and log weights
                for (vector<Label>::iterator lIt = labels.begin(); lIt != labels.end(); ++lIt )
                {
                    hx = pWeakHypothesis->classify(pData, i, lIt->idx ); // h_l(x_i)
                    _hy[i][lIt->idx] += alpha * hx; // alpha * h_l(x_i)                             
                    lIt->weight -= alpha * hx * lIt->y; // log(exp( -alpha * h_l(x_i) ))
                }
            }
        }


        // centering the log weights for avoiding numerical problems
        AlphaReal meanLogWeights = 0; // the mean of the log weights
        int numLabels = 0; // the number of the labels (should be counted since the label vector is of variable size)
        for (int i = 0; i < numExamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            for (vector<Label>::iterator lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                meanLogWeights += lIt->weight;
                numLabels++;
            }
        }
        meanLogWeights /= numLabels;

        // computing the normalization factor 
        AlphaReal Z = 0; // the normalization factor
        for (int i = 0; i < numExamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            for (vector<Label>::iterator lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                lIt->weight -= meanLogWeights;
                Z += exp(lIt->weight);
            }
        }

        // normalizing and exponentiating the weights
        for (int i = 0; i < numExamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            for (vector<Label>::iterator lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                lIt->weight = exp(lIt->weight)/Z;
            }
        }

        //upload the margins 
        pOutInfo->setTable( pData, _hy );
        pOutInfo->setStartingIteration(numIters);
        return 0;
    }

    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------

    AlphaReal AdaBoostPLLearner::updateWeights(InputData* pData, BaseLearner* pWeakHypothesis)
    {
        const int numExamples = pData->getNumExamples();
        const int numClasses = pData->getNumClasses();

        const AlphaReal alpha = pWeakHypothesis->getAlpha();

        AlphaReal Z = 0; // The normalization factor

        _hy.resize(numExamples);
        for ( int i = 0; i < numExamples; ++i) {
            _hy[i].resize(numClasses);
            fill( _hy[i].begin(), _hy[i].end(), 0.0 );
        }
        // recompute weights
        // computing the normalization factor Z

        // for each example
        for (int i = 0; i < numExamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            vector<Label>::iterator lIt;

            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                _hy[i][lIt->idx] = pWeakHypothesis->classify(pData, i, lIt->idx) * // h_l(x_i)
                    lIt->y;
                Z += lIt->weight * // w
                    exp( 
                            -alpha * _hy[i][lIt->idx] // -alpha * h_l(x_i) * y_i
                       );
                // important!
                // _hy[i] must be a vector with different sizes, depending on the
                // example!
                // so it will become:
                // _hy[i][l] 
                // where l is NOT the index of the label (lIt->idx), but the index in the 
                // label vector of the example
            }
        }

        AlphaReal gamma = 0;

        // Now do the actual re-weight
        // (and compute the edge at the same time)
        // for each example
        for (int i = 0; i < numExamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            vector<Label>::iterator lIt;

            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                AlphaReal w = lIt->weight;
                gamma += w * _hy[i][lIt->idx];
                //if ( gamma < -0.8 ) {
                //      cout << gamma << endl;
                //}
                // The new weight is  w * exp( -alpha * h(x_i) * y_i ) / Z
                lIt->weight = w * exp( -alpha * _hy[i][lIt->idx] ) / Z;
            }
        }


        //for (int i = 0; i < numExamples; ++i)
        //{
        //   for (int l = 0; l < numClasses; ++l)
        //   {
        //      _hy[i][l] = pWeakHypothesis->classify(pData, i, l) * // h_l(x_i)
        //                  pData->getLabel(i, l); // y_i

        //      Z += pData->getWeight(i, l) * // w
        //           exp( 
        //             -alpha * _hy[i][l] // -alpha * h_l(x_i) * y_i
        //           );
        //   } // numClasses
        //} // numExamples

        // The edge. It measures the
        // accuracy of the current weak hypothesis relative to random guessing

        //// Now do the actual re-weight
        //// (and compute the edge at the same time)
        //for (int i = 0; i < numExamples; ++i)
        //{
        //   for (int l = 0; l < numClasses; ++l)
        //   {  
        //      float w = pData->getWeight(i, l);

        //      gamma += w * _hy[i][l];

        //      // The new weight is  w * exp( -alpha * h(x_i) * y_i ) / Z
        //      pData->setWeight( i, l, 
        //                        w * exp( -alpha * _hy[i][l] ) / Z );
        //   } // numClasses
        //} // numExamples

        return gamma;
    }

    // -------------------------------------------------------------------------

    int AdaBoostPLLearner::resumeWeakLearners(InputData* pTrainingData)
    {
        if (_resumeShypFileName.empty())
            return 0;

        if (_verbose > 0)
            cout << "Reloading strong hypothesis file <" << _resumeShypFileName << ">.." << flush;

        // The class that loads the weak hypotheses
        UnSerialization us;

        // loads them
        us.loadHypotheses(_resumeShypFileName, _foundHypotheses, pTrainingData, _verbose);

        if (_verbose > 0)
            cout << "Done!" << endl;

        // return the number of iterations found
        return static_cast<int>( _foundHypotheses.size() );
    }

    // -------------------------------------------------------------------------

    void AdaBoostPLLearner::resumeProcess(Serialization& ss, 
            InputData* pTrainingData, InputData* pTestData, 
            OutputInfo* pOutInfo)
    {

        _earlyStoppingDone = false;
        if (_resumeShypFileName.empty())
            return;

        vector<BaseLearner*>::iterator it;
        int t;

        // rebuild the new strong hypothesis file
        for (it = _foundHypotheses.begin(), t = 0; it != _foundHypotheses.end(); ++it, ++t)
        {
            BaseLearner* pWeakHypothesis = *it;

            // append the current weak learner to strong hypothesis file,
            ss.appendHypothesis(t, pWeakHypothesis);
        }

        if ( _fastResumeProcess ) { // The AdaBost will recalculate of the last iteration based on the margins
            // Updates the weights
            if (_verbose > 0)
                cout << "Recalculating the weights of training data...";

            updateWeights(pOutInfo, pTrainingData, _foundHypotheses);

            if (_verbose > 0)
                cout << "Done" << endl;
            if (pTestData)
            {
                if (_verbose > 0)
                    cout << "Recalculating the weights of test data...";
                updateWeights(pOutInfo, pTestData, _foundHypotheses);
            }
            if (_verbose > 0)
                cout << "Done" << endl;
            // Output the step-by-step information
            //printOutputInfo(pOutInfo, _foundHypotheses.size(), pTrainingData, pTestData, pWeakHypothesis);
        } else { //slow resume process, in this case the AdaBoost will recalculate the error rates of all iterations
            const int numIters = static_cast<int>(_foundHypotheses.size());
            const int step = numIters < 5 ? 1 : numIters / 5;

            OUTPUTINFO_OPTIMIZATION opt_type;
            if (_earlyStopping)
            {
                opt_type = pOutInfo->getOutputInfoObject(_earlyStoppingOutputColumn)->getOptimType();
                if (opt_type == UNKNOWN)
                {
                    cerr << "ERROR!" 
                        << _earlyStoppingOutputColumn 
                        << " cannot be selected for earlyStopping policy " << endl;
                    exit(1);

                }
            }

            if (_verbose > 0)
                cout << "Resuming up to iteration " << _foundHypotheses.size() - 1 << ": 0%." << flush;
            // simulate the AdaBoost algorithm for the weak learners already found
            for (it = _foundHypotheses.begin(), t = 0; it != _foundHypotheses.end(); ++it, ++t)
            {
                BaseLearner* pWeakHypothesis = *it;

                // Output the step-by-step information
                if (pOutInfo)
                    printOutputInfo(pOutInfo, t, pTrainingData, pTestData, pWeakHypothesis);

                // Updates the weights and returns the edge
                AlphaReal gamma = updateWeights(pTrainingData, pWeakHypothesis);

                if (_verbose > 1 && (t + 1) % step == 0)
                {
                    float progress = static_cast<float>(t) / static_cast<float>(numIters) * 100.0;                             
                    cout << "." << setprecision(2) << progress << "%." << flush;
                }

                // If gamma <= theta there is something really wrong.
                if (gamma <= _theta)
                {
                    cerr << "ERROR!" <<  setprecision(4) << endl
                        << "At iteration <" << t << ">, edge smaller than the edge offset (theta). Something must be wrong!" << endl
                        << "[Edge: " << gamma << " < Offset: " << _theta << "]" << endl
                        << "Is the data file the same one used during the original training?" << endl;
                    //          exit(1);
                }

                // Updating earlystopping status for slow resume
                if (_earlyStopping)
                {
                    sumErrorWindow += pOutInfo->getOutputHistory(pTestData, _earlyStoppingOutputColumn, t);
                    numErrorWindow += 1;
                    while (numErrorWindow > _earlyStoppingSmoothingWindowRate * t + 1)
                    {
                        sumErrorWindow -= pOutInfo->getOutputHistory(pTestData, _earlyStoppingOutputColumn, t - numErrorWindow + 1);
                        numErrorWindow -= 1;
                    }
                    if (t > _earlyStoppingMinIterations)
                    {
                        if (((opt_type == MIN) && ((sumErrorWindow / numErrorWindow) < currentMin)) ||
                                ((opt_type == MAX) && ((sumErrorWindow / numErrorWindow) > currentMin)))
                        {
                            currentMin = sumErrorWindow / numErrorWindow;
                            _currentMinT = t;
                        }
                        //						cout << _earlyStoppingMinIterations << " " << t << ": " << sumErrorWindow / numErrorWindow << " " << _currentMinT << endl;
                        if (t > _currentMinT * _earlyStoppingMaxLookaheadRate)
                        {
                            cout << "Early Stopping at " << t << endl;
                            _earlyStoppingDone = true;
                            break;
                        }
                    }
                }


            }  // loop on iterations
        }

        if (_verbose > 0)
            cout << "Done!" << endl;

    }

    // -------------------------------------------------------------------------

    void AdaBoostPLLearner::printOutputInfo(OutputInfo* pOutInfo, int t, 
            InputData* pTrainingData, InputData* pTestData, 
            BaseLearner* pWeakHypothesis)
    {

        pOutInfo->outputIteration(t);
        pOutInfo->outputCustom(pTrainingData, pWeakHypothesis);

        if (pTestData)
        {

            pOutInfo->separator();
            pOutInfo->outputCustom(pTestData, pWeakHypothesis);

        }

        pOutInfo->outputCurrentTime();
        pOutInfo->endLine();
    }

    // -------------------------------------------------------------------------
    void AdaBoostPLLearner::printOutWeights( InputData* pData )
    {
        // if there is no weight file name is given, then it returns
        if ( _weightFile.empty() ) return;

        if ( _verbose > 3 ) cout << "Print out weights file!" << endl;

        ofstream outStream;
        // open the stream
        outStream.open(_weightFile.c_str());

        // is it really open?
        if ( ! outStream.is_open() )
        {
            cerr << "ERROR: cannot open the output stream (<"
                << _weightFile << ">) for output the weights!" << endl;
            exit(1);
        }

        const int numExamples = pData->getNumExamples();
        const int numClasses = pData->getNumClasses();


        // for each example
        for (int i = 0; i < numExamples; ++i)
        {
            vector<Label>& labels = pData->getLabels(i);
            vector<Label>::iterator lIt;

            for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
            {
                outStream << lIt->weight << ";";
            }
            outStream << endl;
        }


        outStream.close();
    }

    // -------------------------------------------------------------------------
    void AdaBoostPLLearner::run( const nor_utils::Args& args, InputData* pTrainingData, const string baseLearnerName, const int numIterations, vector<BaseLearner*>& foundHypotheses )
    {

        // get the registered weak learner (type from name)
        BaseLearner* pWeakHypothesisSource = 
            BaseLearner::RegisteredLearners().getLearner(baseLearnerName);
        // initialize learning options; normally it's done in the strong loop
        // also, here we do it for Product learners, so input data can be created
        pWeakHypothesisSource->initLearningOptions(args);

        BaseLearner* pConstantWeakHypothesisSource = 
            BaseLearner::RegisteredLearners().getLearner("ConstantLearner");


        if (_verbose == 1)
            cout << "Learning in progress... " << flush;


        ///////////////////////////////////////////////////////////////////////
        // Starting the AdaBoost main loop
        ///////////////////////////////////////////////////////////////////////
        for (int t = 0; t < numIterations; ++t)
        {
            if (_verbose > 0)
                cout << (t+1) << ", " << flush;

            BaseLearner* pWeakHypothesis = pWeakHypothesisSource->create();
            pWeakHypothesis->initLearningOptions(args);
            //pTrainingData->clearIndexSet();

            pWeakHypothesis->setTrainingData(pTrainingData);

            AlphaReal energy = pWeakHypothesis->run();

            //float gamma = pWeakHypothesis->getEdge();
            //cout << gamma << endl;

            if ( (_withConstantLearner) || ( energy != energy ) ) // check constant learner if user wants it (if energi is nan, then we chose constant learner
            {
                BaseLearner* pConstantWeakHypothesis = pConstantWeakHypothesisSource->create() ;
                pConstantWeakHypothesis->initLearningOptions(args);
                pConstantWeakHypothesis->setTrainingData(pTrainingData);
                AlphaReal constantEnergy = pConstantWeakHypothesis->run();

                if ( (constantEnergy <= energy) || ( energy != energy ) || ( nor_utils::is_zero(constantEnergy - energy))) {
                    delete pWeakHypothesis;
                    pWeakHypothesis = pConstantWeakHypothesis;
                }
            }

            if (_verbose > 1)
                cout << "Weak learner: " << pWeakHypothesis->getName()<< endl;

            // Updates the weights and returns the edge
            AlphaReal gamma = updateWeights(pTrainingData, pWeakHypothesis);

            if (_verbose > 1)
            {
                cout << setprecision(5)
                    << "--> Alpha = " << pWeakHypothesis->getAlpha() << endl
                    << "--> Edge  = " << gamma << endl
                    << "--> Energy  = " << energy << endl
                    //            << "--> ConstantEnergy  = " << constantEnergy << endl
                    //            << "--> difference  = " << (energy - constantEnergy) << endl
                    ;
            }

            // If gamma <= theta the algorithm must stop.
            // If theta == 0 and gamma is 0, it means that the weak learner is no better than chance
            // and no further training is possible.
            if (gamma <= _theta)
            {
                if (_verbose > 0)
                {
                    cout << "Can't train any further: edge = " << gamma 
                        << " (with and edge offset (theta)=" << _theta << ")" << endl;
                }

                //          delete pWeakHypothesis;
                //          break; 
            }

            // Add it to the internal list of weak hypotheses
            foundHypotheses.push_back(pWeakHypothesis); 

        }  // loop on iterations
        /////////////////////////////////////////////////////////

        if (_verbose > 0)
            cout << "AdaBoost Learning completed." << endl;
    }
    // -------------------------------------------------------------------------
} // end of namespace MultiBoost
