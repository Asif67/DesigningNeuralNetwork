using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DesigningNeuralNetwork.SupervisedLearning.MultipleInput.MultipleHiddenLayer.MultipleNeuron.MultipleOutput
{
    class Output
    {
        public double enteringCostWeight;
        public double activationValue;
        public double bias;
        public double desiredOutput;
        public void ActivationvalueCalculation(double previousActivationValue)
        {
            activationValue = SigmoidActivationFunction((enteringCostWeight * previousActivationValue) + bias);
        }
        public double SigmoidActivationFunction(double x)
        {
            double sig = 1 / (1 + Math.Pow(Math.E, -x));
            return sig;
        }
        public double ErrorCalculation()
        {
            double cost;
            cost = Math.Pow((activationValue - desiredOutput),2);
            return (cost);

        }
    }
}
