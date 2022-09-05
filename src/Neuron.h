#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <random>
#include "Layer.h"

class Layer;

class Neuron
{
private:
    //Weight Randomization members
    static std::random_device rd;
    static std::seed_seq seed;
    static std::mt19937 mt;
    static std::uniform_real_distribution<> randomGenerator;

    //Constants
    static constexpr double bias = 1.0;

    //Math helper methods
    double activation(double input) const;
    double activationPrime(double input) const;
public:
    //Initialization/destruction methods
    Neuron();
    Neuron(int numberOfWeights);
    Neuron(const Neuron& orig);
    ~Neuron();
    //Randomizes the weights of the neuron (if it isnt an input neuron)
    void reinitializeWeights(int weightCount);

    //Member variables
    bool isInputNeuron;
    std::vector<double> inboundWeights;
    double neuronValue;
    std::vector<double> lastWeightChange;

    //Updates this neuron's value given the values of the previous layer
    void update(const Layer& previousLayer);

    //Prints the neuron to console
    void printToConsole() const;

    //Learning methods
    //Find the partial derivative of a Weight linking to this neuron
    double findCostOfWeight(const Layer& previousLayer ,int weightIndex, double derivativeOfCostRespectNeuron) const;
    //Find the partial derivative of a neuron in a "back"/"previous" layer through this neuron
    double findCostOfPrevNeuron(const Layer& previousLayer, int neuronIndex, double derivativeOfCostRespectNeuron) const;
    //Adjusts the weights of this neuron using the partial derivative of the cost function
    void adjustInboundWeights(const Layer& previousLayer, double derivativeOfCostRespectNeuron, double learningRate, double momentumFactor);
    
    //Find the error on this neuron
    double findError(double desiredValue) const;
    //Find the derivative of the error on this neuron
    double findErrorPrime(double desiredValue) const;
};

#endif