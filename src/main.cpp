#include "NeuralNetwork.h"
#include <iostream>
#include <random>
#include <vector>

double scaleValue(double x, double currentMax, double currentMin, double desiredMax, double desiredMin) {
    return (((desiredMax-desiredMin)*(x - currentMin))/(currentMax-currentMin)) + desiredMin;
}

int main() {

    NeuralNetwork newNet{2, 1};
    newNet.update();

    std::random_device rd;
    std::seed_seq seed{rd(),rd(),rd()};
    std::mt19937 mt;
    double distributionMin = -1;
    double distributionMax = 1;
    std::uniform_real_distribution<double> distribution(distributionMin,distributionMax);
    
    double m;
    std::cout << "\nThis is a linear perceptron.\nEnter the m value of the following formula y = m * x + b : ";
    std::cin >> m;
    double b;
    std::cout << "\nEnter the b value of the following formula y = "<<m<<" * x + b : ";
    std::cin >> b;

    
    double minx;
    std::cout << "\nWhat is the minimum value of x for this line?: ";
    std::cin >> minx;
    double maxx;
    std::cout << "\nWhat is the maximum value of x for this line?: ";
    std::cin >> maxx;

    double scaledB = scaleValue(b,maxx,minx,1.0,-1.0);

    while (true) {
        int trainingNum;
        int batchNum;
        std::cout << "\nHow many random training examples would you like to give in each batch?: ";
        std::cin >> batchNum;
        std::cout << "\nHow many batches would you like to give?: ";
        std::cin >> trainingNum;
        if (trainingNum==0 || batchNum==0) {
            break;
        }
        for (int b=0;b<trainingNum;b++) {
            int batchSize =10;
            std::vector<std::vector<double>> inputs;
            std::vector<std::vector<double>> outputs;
            for (int batch=0;batch<batchSize;batch++) {
                double x = distribution(mt);
                double y = distribution(mt);
                inputs.push_back({x,y});
                outputs.push_back({(x*(m) + (scaledB) > y)?1.0:-1.0});
            }
            newNet.trainFromInputSet(inputs,outputs);
        }
        double perceptronM = (newNet.outputLayer.containedNeurons[0].inboundWeights[0] / -newNet.outputLayer.containedNeurons[0].inboundWeights[1]);
        double perceptronB = (newNet.outputLayer.containedNeurons[0].inboundWeights[2] / -newNet.outputLayer.containedNeurons[0].inboundWeights[1]);
        double scaledPerceptronB = scaleValue(perceptronB,1,-1,maxx,minx);
        std::cout << "\nThe perceptron thinks the formula is: y = "<<perceptronM<<" * x + "<<scaledPerceptronB;
    }

    double perceptronM = (newNet.outputLayer.containedNeurons[0].inboundWeights[0] / -newNet.outputLayer.containedNeurons[0].inboundWeights[1]);
    double perceptronB = (newNet.outputLayer.containedNeurons[0].inboundWeights[2] / -newNet.outputLayer.containedNeurons[0].inboundWeights[1]);
    double scaledPerceptronB = scaleValue(perceptronB,1,-1,maxx,minx);
    std::cout << "\nThe perceptron decides that the formula is: y = "<<perceptronM<<" * x + "<<scaledPerceptronB;

    std::cout << "\nPress Enter to exit the program ";
    std::cin.clear(); 
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get(); 

    return 0;
}

