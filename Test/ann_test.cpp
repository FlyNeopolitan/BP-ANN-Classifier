#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "../Classifier/ann_classifier.h"


TEST_CASE("construction") {
    ANNClassifier test;
    for (unsigned i = 0; i < 5; ++i) {
        test.setInput(i);
        std::vector<int> expected(i, 0);
        REQUIRE(test.classify(expected) == expected);
    }
}

TEST_CASE("Reset Neurons") {
    ANNClassifier test;
    test.resetNeurons(std::vector<unsigned>{1,2,3});
    auto actual = test.strutureMatrix();
    std::vector<std::vector<unsigned>> expected{{2}, {3,3}, {0,0,0}};
    REQUIRE(actual ==expected);
}

