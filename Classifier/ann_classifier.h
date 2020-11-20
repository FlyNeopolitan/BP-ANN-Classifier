#pragma once
#include "classifier.h"
#include "math.h"

class ANNClassifier : Classifier {

public:

    /**
    * initilize or reset the input with certain size
    * @param num represents the number of volume of input vector
    * edage case : when there's no layer, we will add a layer as the input layer
    */
    void setInput(const unsigned& num);
    
    /**
    * initilize or reset the output with certain size
    * @param num represents the number of volume of output vector
    * edge case : when there's only one layer(input layer), we will add a layer as a output layer
    * edge case : when there's no layer, do nothing
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
    * when there's only one layer(only input layer), the output will be exactly the same as input
    */
    std::vector<double> classify(const std::vector<double>& input) const;

    /**
    * train the Classifier for a pair of input and output
    * @param input represents an input 
    * @param output represents expected output
    */
    void train(const std::vector<double>& input, const std::vector<double>& output);

    /**
    * clear all the data.
    */
    void clear();
    
    /**
    * @return the structure Matrix of Neurals
    * For example, if there are 2,3,4 Neuronses in level 1,2,3
    * then we will get a vector {{3,3}, {4,4,4}, {0, 0, 0, 0}}.
    */
    std::vector<std::vector<unsigned>> strutureMatrix() const;

    /**
    * @return the biasVector
    */
    std::vector<double> biasVector() const;

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
    double sigmoidFunction(double x, double alpha = 1) const;

    std::vector<std::vector<Neural>> neurals_;
    std::vector<double> bias_;

    const double initial_weight = 1;
    const double initial_bias = 0;
    
};


