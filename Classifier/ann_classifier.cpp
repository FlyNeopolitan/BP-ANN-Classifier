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
    if (neurals_.size() == 1) {
        neurals_.push_back(std::vector<Neural>(num, Neural()));
        for (Neural& neural : neurals_[0]) {
            neural.weight = std::vector<double>(num, initial_weight);
        }
        bias_.push_back(initial_bias);
    }
}


void ANNClassifier::resetNeurons(const std::vector<unsigned>& Neurons) {
    clear();
    for (unsigned i = 0; i < Neurons.size() - 1; ++i) {
        //insert layer at index i 
        std::vector<Neural> current(Neurons[i], 
            Neural(std::vector<double>(Neurons[i + 1], initial_weight)));
        neurals_.push_back(current);
        bias_.push_back(initial_bias);
    }
    //instert last layer
    neurals_.push_back(std::vector<Neural>(Neurons.back(), Neural()));
}


std::vector<double> ANNClassifier::classify(const std::vector<double>& input) const {
    //value represents results for each partition
    auto value = valueMatrix(input);
    //output will be the last layer
    return value[value.size() - 1];
}


void ANNClassifier::train(const std::vector<double>& input, const std::vector<double>& output) {
    //get derivativeMatrix
    auto derivativeMatrix = derivative(input, output);
    //update weights
    for (unsigned layer = 0; layer < neurals_.size() - 1; ++layer) {
        for (unsigned unit = 0; unit < neurals_[layer].size(); ++unit) {
            for (unsigned w = 0; w < neurals_[layer][unit].weight.size(); ++w) {
                neurals_[layer][unit].weight[w] -= learningRate * derivativeMatrix[layer][unit][w];
            }
        }
    }
    //to do : updates bias
}


void ANNClassifier::clear() {
    //to do
    neurals_.clear();
    bias_.clear();
}


double ANNClassifier::activeFunction(double x) const {
    return sigmoidFunction(x);
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


std::vector<double> ANNClassifier::biasVector() const {
    return bias_;
}


std::vector<std::vector<double>> ANNClassifier::valueMatrix(const std::vector<double>& input) const {
    std::vector<std::vector<double>> value;
    value.push_back(input);
    for (unsigned i = 0; i < neurals_.size() - 1; ++i) {
        std::vector<double> current = value[i];
        std::vector<double> next;
        //update value for Jth item of next layer
        for (unsigned j = 0; j < neurals_[i + 1].size(); ++j) {
            double v = 0;
            //use current layer's value and weight to calculate Kth value of Nerual at next layer
            for (unsigned k = 0; k < neurals_[i].size(); ++k) {
                v += current[k] * neurals_[i][k].weight[j];  
            }
            v = sigmoidFunction(v + bias_[i]);
            next.push_back(v);
        }
        value.push_back(next);
    }
    return value;
}


std::vector<std::vector<std::vector<double>>>
    ANNClassifier::derivative(const std::vector<double>& input, const std::vector<double>& output) {
        auto values = valueMatrix(input);
        auto actual = classify(input);
        std::vector<std::vector<std::vector<double>>> results;
        //initilize the matrix
        for (unsigned i = 0; i < neurals_.size() - 1; ++i) {
            results.push_back(std::vector<std::vector<double>>());
        }
        //update last layer
        int lastIndex = results.size() - 1;
        for (unsigned i = 0; i < neurals_[lastIndex].size(); ++i) {
            std::vector<double> current;
            for (unsigned j = 0; j < neurals_[lastIndex][0].weight.size(); ++j) {
                double gradient_E_out = - output[j] + actual[j];
                double gradient_out_net = actual[j] * (1 - actual[j]);
                double gradient_net_w = values[lastIndex][j];
                double d = gradient_E_out * gradient_out_net * gradient_net_w;
                current.push_back(d);
            }
            results[lastIndex].push_back(current);
        }
        //from last to front, update layers
        for (int i = lastIndex - 1; i >= 0; --i) {
            for (unsigned j = 0; j < neurals_[i].size(); ++j) {
                std::vector<double> current;
                for (unsigned k = 0; k < neurals_[i][0].weight.size(); ++k) {
                    double factor = values[i][j] * (1 - values[i + 1][k]);
                    double E = 0;
                    for (unsigned t = 0; t < results[i + 1][k].size(); ++t) {
                        E += results[i + 1][k][t] * neurals_[i + 1][k].weight[t];
                    }
                    current.push_back(E * factor);
                }
                results[i].push_back(current);
            }
        }
        return results;
}


std::vector<double> ANNClassifier::lossVector(const std::vector<double>& input, const std::vector<double>& output) {
    std::vector<double> E;
    auto result = classify(input);
    for (unsigned i = 0; i < result.size(); ++i) {
        double value = (input[i] - output[i]) * (input[i] - output[i]) / 2;
        E.push_back(value);
    }
    return E;
}




