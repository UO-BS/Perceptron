#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Layer.h"
#include <vector>

class NeuralNetwork
{
private:

    //Updates all the weights in the network a vector<vector<double>> of neuron cost partial derivatives
    void updateWeightsFromCost(const std::vector<std::vector<double>>& cost);
    //Finds the partial derivative of the cost function for every neuron in the network
    std::vector<std::vector<double>> getNeuronCosts(const std::vector<double>& desiredValues) const;

public:

    Layer inputLayer;
    std::vector<Layer> hiddenLayers;
    Layer outputLayer;
    double learningRate;
    double momentumFactor;

    NeuralNetwork() = delete;
    NeuralNetwork(int inputLayerSize, int outputLayerSize, double learningRate = 0.1, double momentumFactor = 0.0);
    NeuralNetwork(const NeuralNetwork& orig);
    ~NeuralNetwork();

    //Adds a new hidden layer into the network (layer is added right before output layer)
    void addHiddenLayer(int layerSize);

    //Prints the entire network to console
    void printToConsole() const;
    
    //Sets the input neurons
    void setInputNeurons(const std::vector<double>& values);
    //Updates all neurons in the network
    void update();
    //Returns the values of the output neurons
    std::vector<double> getOutputValues();

    //Trains the current network with a set of desired values (input neurons must be set beforehand)
    void train(const std::vector<double>& desiredValues);
    //Trains the network using a training example
    void trainFromInput(const std::vector<double>& inputs, const std::vector<double>& desiredValues);
    //Trains the network using a training set
    void trainFromInputSet(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& desiredValues);

    //Finds the average error on a training set
    double averageErrorOnSet(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& desiredValues);
};

#endif