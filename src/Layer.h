#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "Neuron.h"

class Neuron;

class Layer
{
private:

public:
    Layer() = delete;
    Layer(int layerSize); //For input layer
    Layer(const Layer& previousLayer, int layerSize);
    Layer(std::vector<Neuron> neurons);
    ~Layer();

    std::vector<Neuron> containedNeurons;

    void reassignNeuronsPreviousLayer(const Layer& previousLayer);

    void updateNeurons(const Layer& previousLayer);

    int size() const;
    void printToConsole() const;

    double findCostOfPrevNeuronForLayer(const Layer& previousLayer, int neuronIndex, std::vector<double> derivativeOfCostRespectNeurons) const;

    void adjustContainedNeuronWeights(const Layer& previousLayer, std::vector<double> derivativeOfCostRespectNeurons);
};

#endif