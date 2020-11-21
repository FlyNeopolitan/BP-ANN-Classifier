#include "Classifier/classifier.h"
#include "Classifier/ann_classifier.h"
#include "iostream"
#include <fstream>
#include <sstream>  
#include <string>  
#include <unordered_map>

//Helper method to print Vector
void printVector(std::vector<double> vec);

//Helper method to convert Iris name to index. 
//Iris-virginica : 1, Iris-versicolor : 2, Iris-setosa : 3
double fromNameToIndex(std::string name);

//helper method to convert Iris Index to vector. For example, 0 means {1,0,0}, 1 means {0,1,0}
std::vector<double> fromIndexToVector(double x);

//helper method to determine if two double vector equals. m means margin
bool estimatedEqual(std::vector<double> a, std::vector<double> b, double m = 0.1);

//helper class
template<typename K>
class Four {
public :
    Four(const K& f, const K& s, const K& t, const K& fo) {
        first = f;
        second = s;
        third = t;
        fourth = fo;
    }

    Four() {
        //nothing
    }
    
    bool operator== (const Four<K>& other) const {
        return first == other.first && second == other.second && third == other.third && fourth == other.fourth;
    }
 
    K first;
    K second;
    K third;
    K fourth;
};

//hash function for class Four
size_t FourHash(const Four<double>& p ) 
{
    return std::hash<double>()(p.first) + std::hash<double>()(p.second) + 
        std::hash<double>()(p.third) + std::hash<double>()(p.fourth);
}

//helper method to read in data from IrisData (from UCI Machine Learning Repository)
//it will return a map whose key is four properties of Iris and value is Iris index
auto FileToMap(std::string fileName) {
    std::ifstream Iris(fileName);
    std::unordered_map<Four<double>, double, decltype(&FourHash)> actual(200, FourHash);
    if (Iris.is_open()) {
        std::string str;
        //read in lines
        while (getline(Iris, str)) {
            std::istringstream single(str);
            double data; 
            char comma;
            //name of current Iris
            std::string name;
            //contain four properties of current Iris
            std::vector<double> arr;
            int cnt = 0;
            while (single >> data) {
                if (cnt == 4) {
                    single >> comma;
                    single >> name;
                    break;
                } 
                arr.push_back(data);
                single >> comma;
                cnt++;
            }
            //current Iris properties' vector
            Four<double> four(arr[0], arr[1], arr[2], arr[3]);
            actual[four] = fromNameToIndex(name);
        }
    }
    return actual;
}

//example of using ANNClassifier to classify Iris (data set from UCI Machine Learning Repository)

int main() {
    //an ANNClassifier with 4,10,25,30,25,18,3 Neuronses at each layer, 
    //from input layer to output layer (including)
    ANNClassifier test{4, 10, 25, 30, 25, 18, 3};
    //say hi~
    std::cout << "Hello I am Mr.Classifier" << std::endl;
    //read in Iris
    auto actual = FileToMap("data/iris.txt");
    auto trainning = FileToMap("data/train.txt");
    //train
    for (auto data : trainning) {
        std::vector<double> currentData{data.first.first, data.first.second, data.first.third, data.first.fourth};
        test.train(currentData, fromIndexToVector(data.second));
    }
    //calculate rate
    unsigned cnt = 0;
    unsigned correct = 0;
    for (auto actualData : actual) {
        auto result = test.classify(std::vector<double>{actualData.first.first, 
            actualData.first.second, actualData.first.third, actualData.first.fourth});
        if (estimatedEqual(result, fromIndexToVector(actualData.second))) {
             ++correct;
        }
        cnt++;
    }
    std::cout << "Classifiction Correct Ratio: " << 1.0 * correct / cnt << std::endl;
}


double fromNameToIndex(std::string name) {
    if (name == "Iris-virginica") {
        return 1;
    } else if (name == "Iris-versicolor") {
        return 2;
    } else {
        return 3;
    }
}


std::vector<double> fromIndexToVector(double x) {
    std::vector<double> results(3, 0);
    results[x - 1] = 1;
    return results;
}


bool estimatedEqual(std::vector<double> a, std::vector<double> b, double m) {
    for (unsigned i = 0; i < a.size(); ++i) {
        if (abs(a[i] - b[i]) > m) {
            return false;
        }
    }
    return true;
}


void printVector(std::vector<double> vec) {
    for (auto i : vec) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}







