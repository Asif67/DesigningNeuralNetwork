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
        public void ActivationvalueCalculation(double previousActivationValue)
        {
            activationValue = SigmoidActivationFunction((enteringCostWeight * previousActivationValue) + bias);
        }
        public double SigmoidActivationFunction(double x)
        {
            double sig = 1 / (1 + Math.Pow(Math.E, -x));
            return sig;
        }
        
    }
}
