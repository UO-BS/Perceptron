#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Layer.h"
#include <vector>

class NeuralNetwork
{
private:

    

public:

    Layer inputLayer;
    std::vector<Layer> hiddenLayers;
    Layer outputLayer;

    NeuralNetwork() = delete;
    NeuralNetwork(int inputLayerSize, int outputLayerSize);
    ~NeuralNetwork();

    void addHiddenLayer(int layerSize);

    void printToConsole() const;

    void setInputNeurons(const std::vector<double>& values);
    void update();
    std::vector<double> getOutputValues();

    void train(std::vector<double> desiredValues);
};

#endif