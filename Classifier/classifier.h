#pragma once
#include <vector>

class Classifier {
    
    /**
    * initilize or reset the input with certain size
    * @param num represents the number of volume of input vector
    */
    virtual void setInput(const unsigned&  num) = 0;
    
    /**
    * initilize or reset the output with certain size
    * @param num represents the number of volume of output vector
    */
    virtual void setOutput(const unsigned&  num) = 0; 
    
    /**
    * @param input represents the input vector
    * @return the output vector
    */
    virtual std::vector<double> classify(const std::vector<double>& input) const = 0;

    /**
    * train the Classifier for a pair of input and output
    * @param input represents an input 
    * @param output represents expected output
    */
    virtual void train(const std::vector<double>& input, const std::vector<double>& output) = 0;

    /**
    * clear all the data.
    */
    virtual void clear() = 0;

};
