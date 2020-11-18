#pragma once
#include "classifier.h"

class ANNClassifier : Classifier {

public:

    /**
    * initilize or reset the input with certain size
    * @param num represents the number of volume of input vector
    */
    void setInput(const unsigned& num);
    
    /**
    * initilize or reset the output with certain size
    * @param num represents the number of volume of output vector
    */
    void setOutput(const unsigned& num);
    
    /**
    * reset all the Neurons
    * @param Neurons represents the vector of Neurons, from input to output
    * for example, vector {2,3,2} indicates there are 2,3,2 Neuronses in level 1,2,3. 
    * (Level 1 is closest to input layer, and level 3 is closest to output layer)
    */
    void resetNeurons(const std::vector<unsigned>& Neurons);

    /**
    * classify the input
    * @param input represents the input vector
    * @return the output vector
    */
    std::vector<int> classify(const std::vector<int>& input) const;

    /**
    * train the Classifier for a pair of input and output
    * @param input represents an input 
    * @param output represents expected output
    */
    void train(const std::vector<int>& input, const std::vector<int>& output);

    /**
    * clear all the data.
    */
    void clear();

private:

    class Neural {
        public:   
        //constructor for inner class
        Neural(std::vector<double> weight) {
            this->weight = weight;
        }

        //constructor for inner class
        Neural() {
            //nothing
        }
        
        std::vector<double> weight;
    };
    
    /**
     * sigmoid Function
     * @param x represents input
     * @return f(x) where f is sigmoid Function
     */
    int sigmoidFunction(double x) const;

    std::vector<std::vector<Neural>> neurals_;
    const int initial_weight = 1;
    
};


