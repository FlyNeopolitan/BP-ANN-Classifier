#include "ann_classifier.h"

void ANNClassifier::setInput(const unsigned& num) {
    if (neurals_.size() == 0) {
        //create and insert the input Neural
        neurals_.push_back(std::vector<Neural>(num, Neural()));
    } else if (neurals_.size() == 1) {
        //change the input Neural, and it is also the output layer
        neurals_[0] = std::vector<Neural>(num, Neural());
    } else {
        //change the input Neural, and adjust it with the second layer with initial_weight
        neurals_[0] = std::vector<Neural>(num, 
            Neural(std::vector<double>(neurals_[1].size(), initial_weight)));
    }
}


void ANNClassifier::setOutput(const unsigned& num) {
    if (neurals_.size() > 1) {
        //change output layout
        neurals_[neurals_.size() - 1] = std::vector<Neural>(num, Neural());
        //change last layout
        for (Neural& neural : neurals_[neurals_.size() - 2]) {
            neural.weight = std::vector<double>(num, initial_weight);
        }
    }
}


void ANNClassifier::resetNeurons(const std::vector<unsigned>& Neurons) {
    for (unsigned i = 0; i < Neurons.size() - 1; ++i) {
        //insert layer at index i 
        std::vector<Neural> current(Neurons[i], 
            Neural(std::vector<double>(Neurons[i + 1], initial_weight)));
        neurals_.push_back(current);
    }
    //instert last layer
    neurals_.push_back(std::vector<Neural>(Neurons.back(), Neural()));
}


std::vector<int> ANNClassifier::classify(const std::vector<int>& input) const {
    //value represents results for each partition
    std::vector<std::vector<int>> value;
    value.push_back(input);
    for (unsigned i = 0; i < neurals_.size() - 1; ++i) {
        std::vector<int> current = value[i];
        std::vector<int> next;
        //update value for Jth item of next layer
        for (unsigned j = 0; j < neurals_[i + 1].size(); ++j) {
            int v = 0;
            //use current layer's value and weight to calculate Kth value of Nerual at next layer
            for (unsigned k = 0; k < neurals_[i].size(); ++k) {
                v += sigmoidFunction(current[k] * neurals_[i][k].weight[j]);  
            }
            next.push_back(v);
        }
        value.push_back(next);
    }
    //output will be the last layer
    return value[value.size() - 1];
}


void ANNClassifier::train(const std::vector<int>& input, const std::vector<int>& output) {
    //to do
}

void ANNClassifier::clear() {
    //to do
    neurals_.clear();
}


int ANNClassifier::sigmoidFunction(double x) const {
    return 0;
}

std::vector<std::vector<unsigned>> ANNClassifier::strutureMatrix() const {
    std::vector<std::vector<unsigned>> structure;
    for (unsigned i = 0; i < neurals_.size(); ++i) {
        std::vector<unsigned> currentLayer;
        for (auto neural : neurals_[i]) {
            currentLayer.push_back(neural.weight.size());
        }
        structure.push_back(currentLayer);
    }
    return structure;
}

