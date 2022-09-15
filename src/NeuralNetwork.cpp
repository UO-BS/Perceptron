#include "NeuralNetwork.h"
#include <math.h>
#include <iostream>

NeuralNetwork::NeuralNetwork(int inputLayerSize, int outputLayerSize, double learningRate, double momentumFactor) : 
                                                                        inputLayer{inputLayerSize}, 
                                                                        outputLayer{inputLayer, outputLayerSize},
                                                                        learningRate{learningRate},
                                                                        momentumFactor{momentumFactor}{}
NeuralNetwork::NeuralNetwork(const NeuralNetwork& orig) :   inputLayer{orig.inputLayer}, 
                                                            outputLayer{orig.outputLayer}, 
                                                            hiddenLayers{orig.hiddenLayers},
                                                            learningRate{orig.learningRate},
                                                            momentumFactor{orig.momentumFactor} {}
NeuralNetwork::~NeuralNetwork()
{

}

void NeuralNetwork::addHiddenLayer(int layerSize)
{
    if (hiddenLayers.size() ==0) {  //This is the first hidden layer added
        Layer newLayer = Layer(inputLayer,layerSize);
        hiddenLayers.push_back(newLayer);
        outputLayer.reassignNeuronsPreviousLayer(newLayer);
    } else {                        //There are already hidden layers in this network
        Layer newLayer = Layer(hiddenLayers[hiddenLayers.size()-1],layerSize);
        hiddenLayers.push_back(newLayer);
        outputLayer.reassignNeuronsPreviousLayer(newLayer);
    }
    update();
}

void NeuralNetwork::setInputNeurons(const std::vector<double>& values){
    if (values.size() != inputLayer.size()) {
        //THIS SHOULD RETURN AN ERROR
        return; 
    }
    for (int i=0;i<inputLayer.size();i++) {
        inputLayer.containedNeurons[i].neuronValue = values[i];
    }

    update();
}

void NeuralNetwork::update()
{
    if (hiddenLayers.size()>=1) {
        hiddenLayers[0].updateNeurons(inputLayer);
        for (int i=1;i<hiddenLayers.size();i++) {
            hiddenLayers[i].updateNeurons(hiddenLayers[i-1]);
        }
        outputLayer.updateNeurons(hiddenLayers[hiddenLayers.size()-1]);
    } else {
        outputLayer.updateNeurons(inputLayer);
    }
}

std::vector<double> NeuralNetwork::getOutputValues(){
    std::vector<double> values;
    values.reserve(outputLayer.size());
    for (int i=0;i<outputLayer.size();i++) {
        values.push_back(outputLayer.containedNeurons[i].neuronValue);
    }
    return values;
}

void NeuralNetwork::printToConsole() const
{
    std::cout << "\nNeural Network: \n";
    inputLayer.printToConsole();
    for (int i=0;i<hiddenLayers.size();i++) {
        hiddenLayers[i].printToConsole();
    }
    outputLayer.printToConsole();
    std::cout << "\n";
}

void NeuralNetwork::train(const std::vector<double>& desiredValues)
{
    //Finding the costs
    std::vector<std::vector<double>> derivativeOfCostRespectNeuron = getNeuronCosts(desiredValues);

    //adjusting weights
    updateWeightsFromCost(derivativeOfCostRespectNeuron);
    
    update();
    return;
}

void NeuralNetwork::updateWeightsFromCost(const std::vector<std::vector<double>>& cost)
{
    for (int i=0;i<hiddenLayers.size();i++) {
        if (i==0) {
            hiddenLayers[i].adjustContainedNeuronWeights(inputLayer, cost[i], learningRate, momentumFactor);
        } else {
            hiddenLayers[i].adjustContainedNeuronWeights(hiddenLayers[i-1], cost[i], learningRate, momentumFactor);
        }
    }
    if (hiddenLayers.size() == 0) {
        outputLayer.adjustContainedNeuronWeights(inputLayer, cost[0], learningRate, momentumFactor);
    } else {
        outputLayer.adjustContainedNeuronWeights(hiddenLayers[hiddenLayers.size()-1], cost[cost.size()-1], learningRate, momentumFactor);
    }
}

std::vector<std::vector<double>> NeuralNetwork::getNeuronCosts(const std::vector<double>& desiredValues) const
{
    //Vector of vectors to hold neuron cost partial derivatives ([layer][neuron])
    std::vector<std::vector<double>> derivativeOfCostRespectNeuron(hiddenLayers.size()+1);
    for (int i=0;i<hiddenLayers.size();i++) {
        derivativeOfCostRespectNeuron[i].resize(hiddenLayers[i].size());
    }
    derivativeOfCostRespectNeuron[hiddenLayers.size()].resize(outputLayer.size());

    //Finds cost of each neuron in outputLayer
    for (int i=0;i<outputLayer.size();i++) {
        derivativeOfCostRespectNeuron[hiddenLayers.size()][i] = outputLayer.containedNeurons[i].findErrorPrime(desiredValues[i]);
    }

    //Find cost of last hidden layer (the one that leads into the output) if it exists
    if (hiddenLayers.size()!=0) {
        for (int n=0;n<hiddenLayers[hiddenLayers.size()-1].size();n++) {
            derivativeOfCostRespectNeuron[hiddenLayers.size()-1][n] = outputLayer.findCostOfPrevNeuronForLayer(hiddenLayers[hiddenLayers.size()-1],n,derivativeOfCostRespectNeuron[hiddenLayers.size()]);
        }
    }
    
    //Finds cost of each neuron in hidden layers (excluding the last) if they exist
    for (int l=hiddenLayers.size()-2;l>=0;l--) {
        for (int n=0;n<hiddenLayers[l].size();n++) {
            derivativeOfCostRespectNeuron[l][n] = hiddenLayers[l+1].findCostOfPrevNeuronForLayer(hiddenLayers[l],n,derivativeOfCostRespectNeuron[l+1]);
        }
    }

    return derivativeOfCostRespectNeuron;
}

void NeuralNetwork::trainFromInput(const std::vector<double>& inputs, const std::vector<double>& desiredValues)
{
    if (inputs.size() == 0) {
        //SHOULD THROW AN ERROR
        std::cout << "trainFromInput Error inputs.size() == 0";
        return;
    }

    setInputNeurons(inputs);
    train(desiredValues);
}

void NeuralNetwork::trainFromInputSet(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& desiredValues)
{
    if (inputs.size() != desiredValues.size()) {
        //SHOULD THROW AN ERROR
        std::cout << "trainFromInputSet Error inputs.size() != desiredValues.size()";
        return;
    }
    if (inputs.size() == 0) {
        //SHOULD THROW AN ERROR
        std::cout << "trainFromInputSet Error inputs.size() == 0";
        return;
    }

    int numberOfSets = inputs.size(); //Number of training sets

    //Vector to hold average of the cost partial derivatives on each neuron
    std::vector<std::vector<double>> averageNeuronCosts(hiddenLayers.size()+1);
    for (int i=0;i<hiddenLayers.size();i++) {
        averageNeuronCosts[i].resize(hiddenLayers[i].size());
    }
    averageNeuronCosts[hiddenLayers.size()].resize(outputLayer.size());

    //adding all the costs
    for (int i=0;i<numberOfSets;i++) {
        setInputNeurons(inputs[i]);
        std::vector<std::vector<double>> tempNeuronCosts = getNeuronCosts(desiredValues[i]);
        for (int l=0;l<tempNeuronCosts.size();l++) {
            for (int n=0;n<tempNeuronCosts[l].size();n++) {
                averageNeuronCosts[l][n] += tempNeuronCosts[l][n];
            }
        }
    }
    //averaging the costs
    for (int l=0;l<averageNeuronCosts.size();l++) {
        for (int n=0;n<averageNeuronCosts[l].size();n++) {
            averageNeuronCosts[l][n] /= static_cast<double>(numberOfSets);
        }
    }

    updateWeightsFromCost(averageNeuronCosts);
    update();
}

double NeuralNetwork::averageErrorOnSet(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& desiredValues)
{
    if (inputs.size() != desiredValues.size()) {
        //SHOULD THROW AN ERROR
        std::cout << "trainFromInputSet Error inputs.size() != desiredValues.size()";
        return -1.0;
    }
    if (inputs.size() == 0) {
        //SHOULD THROW AN ERROR
        std::cout << "trainFromInputSet Error inputs.size() == 0";
        return -1.0;
    }

    double error;
    for (int s=0;s<inputs.size();s++) {
        setInputNeurons(inputs[s]);
        for (int n=0;n<desiredValues[s].size();n++) {
            error += outputLayer.containedNeurons[n].findError(desiredValues[s][n]);
        }
    }
    return (error/inputs.size());
}
