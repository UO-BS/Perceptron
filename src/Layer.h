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
    
    //Reinitialize the weights of all neurons in this layer
    void reassignNeuronsPreviousLayer(const Layer& previousLayer);
    
    //Updates all neurons in this layer
    void updateNeurons(const Layer& previousLayer);

    //Returns the number of neurons in this layer
    int size() const;
    //Prints the layer to console 
    void printToConsole() const;

    //Finds the cost of a "back"/"previous" neuron by summing the partial derivative through all neurons in this layer
    double findCostOfPrevNeuronForLayer(const Layer& previousLayer, int neuronIndex, std::vector<double> derivativeOfCostRespectNeurons) const;
    //Adjusts the weights of all neurons in this layer
    void adjustContainedNeuronWeights(const Layer& previousLayer, std::vector<double> derivativeOfCostRespectNeurons);
};

#endif