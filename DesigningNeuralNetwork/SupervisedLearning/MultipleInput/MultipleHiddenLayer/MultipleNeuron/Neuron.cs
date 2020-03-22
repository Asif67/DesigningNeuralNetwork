using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DesigningNeuralNetwork.SupervisedLearning.MultipleInput.MultipleHiddenLayer.MultipleNeuron
{
    class Neuron
    {
        public double enteringCostWeight;
        public double activationValue;
        public double bias;
        public double weightedSum;
        public double x;
        public void ActivationvalueCalculation(double previousActivationValue)
        {
            weightedSum += (enteringCostWeight * previousActivationValue);
        }
        public double SigmoidActivationFunction()
        {
            x = weightedSum + bias;
            activationValue = 1 / (1 + Math.Pow(Math.E, -x));
            return activationValue;
        }
        
    }
}
