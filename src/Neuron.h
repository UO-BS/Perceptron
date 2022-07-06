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
    void reinitializeWeights(int weightCount);

    //Member variables
    bool isInputNeuron;
    std::vector<double> inboundWeights;
    double neuronValue;

    //General Use methods
    void update(const Layer& previousLayer);
    void printToConsole() const;

    //Learning methods
    double findCostOfWeight(const Layer& previousLayer ,int weightIndex, double derivativeOfCostRespectNeuron) const;
    double findCostOfPrevNeuron(const Layer& previousLayer, int neuronIndex, double derivativeOfCostRespectNeuron) const;
    void adjustInboundWeights(const Layer& previousLayer, double derivativeOfCostRespectNeuron);
    double findError(double desiredValue) const;
};

#endif