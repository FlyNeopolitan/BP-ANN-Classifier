#include "ann_classifier.h"

void ANNClassifier::setInput(const unsigned& num) {
    if (neurals_.size() == 0) {
        //create and insert the input Neural
        neurals_.push_back(std::vector<Neural>(num, Neural(true, false)));
    } else if (neurals_.size() == 1) {
        //change the input Neural, and it is also the output layer
        neurals_[0] = std::vector<Neural>(num, Neural(true, true));
    } else {
        //change the input Neural, and adjust it with the second layer with initial_weight
        neurals_[0] = std::vector<Neural>(num, 
            Neural(std::vector<double>(neurals_[1].size(), initial_weight), true, false));
    }
}


void ANNClassifier::setOutput(const unsigned& num) {
    //to do
}


void ANNClassifier::resetNeurons(const std::vector<unsigned>& Neurons) {
    //to do
}


std::vector<int> ANNClassifier::classify(const std::vector<int>& input) const {
    //to do
    return std::vector<int>(0);
}


void ANNClassifier::train(const std::vector<int>& input, const std::vector<int>& output) {
    //to do
}

void ANNClassifier::clear() {
    //to do
}

