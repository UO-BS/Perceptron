#include "Neuron.h"
#include "Layer.h"
#include <math.h>
#include <iostream>

//Randomization variables
std::random_device Neuron::rd;
std::seed_seq Neuron::seed{rd(),rd(),rd()};
std::mt19937 Neuron::mt{seed};
std::uniform_real_distribution<double> Neuron::randomGenerator(-1.0,1.0);

Neuron::Neuron() : isInputNeuron{true}, inboundWeights(0, 0), neuronValue{0}{}
Neuron::Neuron(int numberOfWeights) : isInputNeuron{(numberOfWeights==0)?true : false}, inboundWeights(numberOfWeights + ((numberOfWeights==0)?0 : 1)), neuronValue{0.5}{
    for (int i=0;i<inboundWeights.size();i++) {
        inboundWeights[i] = randomGenerator(mt);
    }

    lastWeightChange.resize(inboundWeights.size(),0);
}
Neuron::Neuron(const Neuron& orig) : isInputNeuron{orig.isInputNeuron}, inboundWeights{orig.inboundWeights}, neuronValue{orig.neuronValue}, lastWeightChange{orig.lastWeightChange}{}
Neuron::~Neuron(){}

void Neuron::reinitializeWeights(int weightCount)
{
    inboundWeights = std::vector<double> (weightCount+ ((weightCount==0)?0 : 1));
    if (weightCount>0) {
        isInputNeuron=false;
    }
    for (int i=0;i<inboundWeights.size();i++) {
        inboundWeights[i] = randomGenerator(mt);
    }

    lastWeightChange.resize(inboundWeights.size(),0);
}

void Neuron::update(const Layer& previousLayer)
{
    if (!isInputNeuron) {
        double temp{};
        temp =0;
        for (int i=0;i<previousLayer.size();i++) {
            temp += previousLayer.containedNeurons[i].neuronValue*inboundWeights[i]; //No bias yet
        }
        temp += inboundWeights[inboundWeights.size()-1]*bias;
        neuronValue = activation(temp);
    }
}

void Neuron::printToConsole() const
{
    if (isInputNeuron) {
        std::cout << "Input Neuron Value: " << neuronValue << "\n";
        return;
    }
    std::cout << "Neuron Value: " << neuronValue << "\n";
    std::cout << "Weights : ";
    for (int i=0;i<inboundWeights.size()-1;i++) {
        std::cout << inboundWeights[i] << " ";
    }
    std::cout << "\nBias ("<<bias<<") Weight: " << inboundWeights[inboundWeights.size()-1];
    std::cout << "\n";
    
}

double Neuron::findCostOfWeight(const Layer& previousLayer, int weightIndex, double derivativeOfCostRespectNeuron) const
{
    //derivative of Cost Function of 1 output with respect to 1 of the Weights
    //Formula: dCost/dActivatedValue * dActivatedValue/dWeightedValue * dWeightedValue/dPrevNeuronValue

    //dActivatedValue/dWeightedValue
    double weightedValue{};
    for (int i=0;i<previousLayer.size();i++) {
        weightedValue += previousLayer.containedNeurons[i].neuronValue*inboundWeights[i];
    }
    weightedValue += bias*inboundWeights[inboundWeights.size()-1];
    double dactivationdLastWeightedValue = activationPrime(weightedValue);
    //dWeightedValue/dPrevNeuronValue
    double dLastWeightedValuedLastValue{};
    if (weightIndex != inboundWeights.size()-1) {   
        dLastWeightedValuedLastValue = previousLayer.containedNeurons[weightIndex].neuronValue;
    } else {    //The weight we are changing is the bias's weight
        dLastWeightedValuedLastValue = bias;
    }

    return  derivativeOfCostRespectNeuron * dactivationdLastWeightedValue * dLastWeightedValuedLastValue;
}

double Neuron::findCostOfPrevNeuron(const Layer& previousLayer, int neuronIndex, double derivativeOfCostRespectNeuron) const
{
    //derivative of Cost Function of 1 output with respect to 1 of the Neurons (previous node)
    //Formula: dCost/dActivatedValue * dActivatedValue/dWeightedValue * dWeightedValue/dInboungWeight
    
    //dActivatedValue/dWeightedValue
    double weightedValue{};
    for (int i=0;i<previousLayer.size();i++) {
        weightedValue += previousLayer.containedNeurons[i].neuronValue*inboundWeights[i];
    }
    weightedValue += bias*inboundWeights[inboundWeights.size()-1];
    double dactivationdLastWeightedValue = activationPrime(weightedValue);
    //dWeightedValue/dInboungWeight
    double dLastWeightedValuedLastWeight = inboundWeights[neuronIndex];

    return derivativeOfCostRespectNeuron * dactivationdLastWeightedValue * dLastWeightedValuedLastWeight;
}

//Neuron activation function
double Neuron::activation(double input) const
{
    return ((exp(input)-exp(-input))/(exp(input)+exp(-input))); // TANH
    //return (1.0 / (1.0 + exp(-input)));   //SIGMOID
}

//Derivative of the neuron activation function (for backpropagation)
double Neuron::activationPrime(double input) const
{
    double activationTemp = activation(input);
    return (1-activationTemp*activationTemp); //TANH PRIME
    //return (activationTemp*(1-activationTemp)); //SIGMOID PRIME
}

void Neuron::adjustInboundWeights(const Layer& previousLayer, double derivativeOfCostRespectNeuron, double learningRate, double momentumFactor) {
    //Vector to hold the required weight changes
    std::vector<double> neededWeightChanges(inboundWeights.size());
    
    //Changing a weight will change the cost function for other weights. The calculation must be seperated from the changes
    
    //Determining how much the weights should change
    for (int j=0;j<inboundWeights.size();j++) {
        neededWeightChanges[j] = (findCostOfWeight(previousLayer, j, derivativeOfCostRespectNeuron)+(momentumFactor*lastWeightChange[j]))*learningRate; 
        
    }
    //Changing the weights
    for (int j=0;j<inboundWeights.size();j++) {
        //Note: i am doing += here and not -= because the derivative of the cost function is negative (but i kept it positive when calculating it)
        double changeValue = neededWeightChanges[j];
        inboundWeights[j] +=  changeValue;
        lastWeightChange[j] = changeValue;
    }

}

double Neuron::findError(double desiredValue) const 
{
    //mean squared error
    double temp = desiredValue - neuronValue;
    return (temp) * (temp);
}

double Neuron::findErrorPrime(double desiredValue) const 
{
    //mean squared error
    return 2*(desiredValue - neuronValue);
}