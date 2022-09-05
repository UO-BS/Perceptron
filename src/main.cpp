#include "NeuralNetwork.h"
#include <iostream>
#include <random>
#include <vector>
#include <math.h>
#include <string>

//Scales the double x, from a range currentMin->currentMax to the range desiredMin->desiredMax
double scaleValue(double x, double currentMax, double currentMin, double desiredMax, double desiredMin) {
    return (((desiredMax-desiredMin)*(x - currentMin))/(currentMax-currentMin)) + desiredMin;
}

//Gets a double value from the user
double getDoubleFromConsole(std::string message, std::string invalidMessage) {
    
    while (true) {
        std::cout << message;
        double temp;
        std::cin >> temp;
        if (!std::cin) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << invalidMessage;
        } else {
            return temp;
        }
    }
}

int main() {
    
    //Getting equation from user
    std::cout << "\nThis is a perceptron. Try to ensure that there is an equal chance of a point being above the curve and below the curve";
    std::cout << "\nIf the difference is too large, the data will be too imbalanced. Consider training in a different domain/range";

    std::string failMessage{"\nInvalid Entry, try again"};
    double mVal = getDoubleFromConsole("\nEnter the m value of the following formula y = m * x^a  + b : ",failMessage);
    double aVal = getDoubleFromConsole("\nEnter the a value of the following formula y = "+std::to_string(mVal)+" * x^a + b : ",failMessage);
    double bVal = getDoubleFromConsole("\nEnter the b value of the following formula y = "+std::to_string(mVal)+" * x^"+std::to_string(aVal)+" + b : ",failMessage);

    //The perceptron's neural network
    NeuralNetwork newNet{2,1,0.25,0.9};
    if (aVal != 1) {
        newNet.addHiddenLayer(3);
    }
    
    //Getting domain and range from user
    std::cout << "\nNote: The perceptron will only train within a certain domain and range (make sure the line crosses this section)";
    double minx = getDoubleFromConsole("\nWhat is the minimum value of x and y for this line? (-1 is recommended): ", failMessage);
    double maxx = getDoubleFromConsole("\nWhat is the maximum value of x and y for this line? (1 is recommended): ", failMessage);

    //Randomization objects for the training sets
    std::random_device rd;
    std::seed_seq seed{rd(),rd(),rd()};
    std::mt19937 mt;
    std::uniform_real_distribution<double> distribution(minx,maxx);

    while (true) {
        int batchNum = getDoubleFromConsole("\nHow many random training examples would you like to give in each batch? (0 will exit training): ", failMessage);
        if (batchNum==0) {break;}
        int trainingNum = getDoubleFromConsole("\nHow many batches would you like to give? (0 will exit training): ",failMessage);
        if (trainingNum==0) {break;}

        //To keep track of average error on the last few batches (to tell the user)
        double accumulatedError=0.0;

        //Training
        for (int b=0;b<trainingNum;b++) {
            std::vector<std::vector<double>> inputs;
            std::vector<std::vector<double>> outputs;
            for (int batch=0;batch<batchNum;batch++) {
                double x = distribution(mt);
                double y = distribution(mt);
                inputs.push_back({scaleValue(x,minx,maxx,-1,1),scaleValue(y,minx,maxx,-1,1)});
                outputs.push_back({(pow(x,aVal)*(mVal) + (bVal) < y)?1.0:-1.0});
            }

            //Keep track of error on the last 10 batches of the training (to tell the user)
            if (b>trainingNum - 10) { //MIGHT BE 11 NOT SURE
                accumulatedError += newNet.averageErrorOnSet(inputs,outputs);
            }

            newNet.trainFromInputSet(inputs,outputs);
        }

        //Tell user average error on last 10 batches (if they chose to use more than 10)
        if (trainingNum<10) {
            accumulatedError /= trainingNum;
        } else {
            accumulatedError /= 10;
        }
        std::cout << "\nThe average error on the last "<<((trainingNum<10)?trainingNum:10)<<" batches is "<<accumulatedError;

    }

    std::cout << "\nTraining is now over.";
    while (true) {
        double x;
        std::cout << "\nEnter the x value of a point to test the network or type any letter to quit: ";
        std::cin >> x;
        if (!std::cin) {
            std::cout << "Invalid Point entered, quitting program";
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cin.clear();
            break;
        }
        double y;
        std::cout << "\nEnter the y value of a point to test the network or type any letter to quit: ";
        std::cin >> y;
        if (!std::cin) {
            std::cout << "Invalid Point entered, quitting program";
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cin.clear();
            break;
        }

        newNet.setInputNeurons({scaleValue(x,minx,maxx,-1,1),scaleValue(y,minx,maxx,-1,1)});
        std::cout << "The Perceptron thinks that the point "<<x<<','<<y<<" is "<<((newNet.getOutputValues()[0]>=0)?"above":"below")<< " the curve/line.";
    }
    
    std::cout << "\nPress Enter to exit the program ";
    std::cin.clear(); 
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get(); 

    return 0;
}

