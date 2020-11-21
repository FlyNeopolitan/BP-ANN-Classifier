#pragma once

#include "classifier.h"
#include "math.h"
#include "../Math/function.h"
#include "iostream"

class ANNClassifier : Classifier {

public:

    /**
    * default constructor : create a empty ANNClassifier
    */
    ANNClassifier();

    /**
    * initialization list constructor
    * @param Neurons represents the vector of Neurons, from input to output
    * for example, vector {2,3,2} indicates there are 2,3,2 Neuronses in level 1,2,3. 
    * (Level 1 is closest to input layer, and level 3 is closest to output layer)
    */
    ANNClassifier(std::initializer_list<unsigned> Neurons);

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

    /**
    * get the derivatives for training for each weight using a 3-dimentional vector
    * @param input represents training input
    * @param output represents expected output 
    * @return derivative Matrix for training
    */
    std::vector<std::vector<std::vector<double>>> derivative(const std::vector<double>& input, const std::vector<double>& output);

    /**
    * reset learning rate to new one
    * @param newRate represents new learning rate
    * set to 1 by default
    */
    void resetLearningRate(double newRate = 1);

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
     * active Function
     * @param x represents input
     * @return f(x) where f is the active function
     * currently we are using sigmoid function with alpha = 1
     */
    double activeFunction(double x) const;
    
    /**
    * @param input represents input vector
    * @return the value Maxtrix according to the input
    */
    std::vector<std::vector<double>> valueMatrix(const std::vector<double>& input) const;

    /**
    * @param input represents trainning input
    * @param output represents expected output
    * @return lossVector of training 
    * lossVector[i] = (output[i-input[i])^2 / output.size()
    */
    std::vector<double> lossVector(const std::vector<double>& input, const std::vector<double>& output);
    
    /**
    * @param derivativeM represents derivativeMatrix
    * @return derivativeBias vector
    */
    std::vector<double> derivativeBiasVector(const std::vector<std::vector<std::vector<double>>>& derivativeM, 
        const std::vector<std::vector<double>>& valueM) const;
    
    //neurals and bias 
    std::vector<std::vector<Neural>> neurals_;
    std::vector<double> bias_;
    //basic setting for learning
    const double initial_weight = 1;
    const double initial_bias = 0;
    double learningRate = 1;
    
};

