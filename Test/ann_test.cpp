#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "../Classifier/ann_classifier.h"

bool estimatedEqual(std::vector<double>, std::vector<double>, double);

TEST_CASE("Set Input") {

    SECTION("layer size = 0 : construction") {
        for (unsigned i = 1; i < 5; ++i) {
            ANNClassifier test;
            test.setInput(i);
            std::vector<double> expectd(i, 0);
            REQUIRE(test.classify(expectd)== expectd);
            REQUIRE(test.biasVector().empty());
        }
    }

    SECTION("layer size = 1 : adjust") {
        ANNClassifier test;
        for (unsigned i = 0; i < 5; ++i) {
        test.setInput(i);
        std::vector<double> expected(i, 0);
        REQUIRE(test.classify(expected) == expected);
        REQUIRE(test.biasVector().empty());
        }
    }

    SECTION("layer size > 1 : adjust") {
        ANNClassifier test;
        test.resetNeurons(std::vector<unsigned>{2,2,3});
        test.setInput(5);
        std::vector<std::vector<unsigned>> expected{{2,2,2,2,2}, {3,3}, {0,0,0}};
        std::vector<double> expectedBias{0, 0};
        REQUIRE(test.strutureMatrix() == expected);
        REQUIRE(test.biasVector() == expectedBias);    
    }
}


TEST_CASE("Reset Neurons") {

    SECTION("initialize") {
        ANNClassifier test;
        test.resetNeurons(std::vector<unsigned>{1,2,3});
        auto actual = test.strutureMatrix();
        std::vector<std::vector<unsigned>> expected{{2}, {3,3}, {0,0,0}};
        REQUIRE(actual ==expected);
        std::vector<double> expectedBias{0, 0};
        REQUIRE(test.biasVector() == expectedBias);   
    }

    SECTION("reset") {
        ANNClassifier test;
        test.resetNeurons(std::vector<unsigned>{3,4,5,2});
        test.resetNeurons(std::vector<unsigned>{1,2,3});
        auto actual = test.strutureMatrix();
        std::vector<std::vector<unsigned>> expected{{2}, {3,3}, {0,0,0}};
        REQUIRE(actual ==expected);
        std::vector<double> expectedBias{0, 0};
        REQUIRE(test.biasVector() == expectedBias);  
    }
}


TEST_CASE("Set Output") {

    SECTION("size > 1 : adjust the output layer") {
        ANNClassifier test;
        test.resetNeurons(std::vector<unsigned>{4,2,5,2});
        test.setOutput(3);
        auto actual = test.strutureMatrix();
        std::vector<std::vector<unsigned>> expected{{2,2,2,2}, {5,5}, {3,3,3,3,3}, {0,0,0}};
        REQUIRE(actual ==expected);
        std::vector<double> expectedBias{0, 0, 0};
        REQUIRE(test.biasVector() == expectedBias);  
    }

    SECTION("size = 1 : add output layer") {
        ANNClassifier test;
        test.resetNeurons(std::vector<unsigned>{4});
        test.setOutput(3);
        auto actual = test.strutureMatrix();
        std::vector<std::vector<unsigned>> expected{{3,3,3,3}, {0,0,0}};
        REQUIRE(actual ==expected);
        std::vector<double> expectedBias{0};
        REQUIRE(test.biasVector() == expectedBias);  
    }

    SECTION("size = 0 : do nothing") {
        ANNClassifier test;
        test.setOutput(3);
        auto actual = test.strutureMatrix();
        REQUIRE(actual.empty());
        REQUIRE(test.biasVector().empty());
    }
}

TEST_CASE("Clear") {
    
    SECTION("empty") {
        ANNClassifier test;
        test.clear();
        auto actual = test.strutureMatrix();
        REQUIRE(actual.empty());
        REQUIRE(test.biasVector().empty());
    }

    SECTION("non-empty") {
        ANNClassifier test;
        test.resetNeurons(std::vector<unsigned>{4,2,5,2});
        test.clear();
        auto actual = test.strutureMatrix();
        REQUIRE(actual.empty());
        REQUIRE(test.biasVector().empty());
    }
}

TEST_CASE("Classify") {

    SECTION("size = 1 : no output layer") {
        ANNClassifier test;
        test.setInput(4);
        std::vector<double> input{3,2,6,1};
        REQUIRE(test.classify(input) == input);
    }

    SECTION("size > 1 : test with small sample 1") {
        ANNClassifier test;
        test.resetNeurons(std::vector<unsigned>{4,2});
        std::vector<double> expected{0.5, 0.5};
        REQUIRE(test.classify(std::vector<double>{0,0,0,0}) == expected);
    }

    SECTION("size > 1 : test with small sample 2") {
        ANNClassifier test;
        test.resetNeurons(std::vector<unsigned>{3,2,3});
        std::vector<double> expected{0.75, 0.75, 0.75};
        REQUIRE(test.classify(std::vector<double>{0,0,0}) == expected);
    }
}


TEST_CASE("Train") {
    SECTION("does train work? : simple test1") {
        ANNClassifier test;
        test.resetNeurons(std::vector<unsigned>{2,1});
        for (unsigned i = 0; i < 1000; ++i) {
            test.train(std::vector<double>{0.5, 0.5}, std::vector<double>{0.9});
        }
        REQUIRE(estimatedEqual(test.classify(std::vector<double>{0.5, 0.5}), std::vector<double>{0.9}, 0.05));
    }

    SECTION("does train work? : simple test2") {
        ANNClassifier test;
        test.resetNeurons(std::vector<unsigned>{3, 3, 2});
        for (unsigned i = 0; i < 1000; ++i) {
            test.train(std::vector<double>{0.2, 0.3, 0.4}, std::vector<double>{0.4, 0.7});
        }
        REQUIRE(estimatedEqual(test.classify(std::vector<double>{0.2, 0.3, 0.4}), std::vector<double>{0.4, 0.7}, 0.05));
    }

    SECTION("does train work? : simple test3") {
        ANNClassifier test;
        test.resetNeurons(std::vector<unsigned>{4, 4, 2, 5, 6, 3});
        for (unsigned i = 0; i < 1000; ++i) {
            test.train(std::vector<double>{0.2, 0.3, 0.4, 0.5}, std::vector<double>{0.1, 0.2, 0.8});
        }
        REQUIRE(estimatedEqual(test.classify(std::vector<double>{0.2, 0.3, 0.4, 0.5}), std::vector<double>{0.1, 0.2, 0.8}, 0.05));
    }
}



bool estimatedEqual(std::vector<double> x, std::vector<double> y, double margin) {
    for (unsigned i = 0; i < x.size() && i < y.size(); ++i) {
        if (abs(x[i] - y[i]) > margin) {
            return false;
        }
    }
    return true;
}