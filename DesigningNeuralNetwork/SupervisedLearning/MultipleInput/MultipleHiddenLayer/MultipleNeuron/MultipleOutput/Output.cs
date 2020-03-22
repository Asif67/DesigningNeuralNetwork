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
        double z;
        public void ActivationvalueCalculation(double previousActivationValue)
        {
            z = (enteringCostWeight * previousActivationValue) + bias;
            activationValue = SigmoidActivationFunction(z);
        }
        double SigmoidActivationFunction(double x)
        {
            double sig = 1 / (1 + Math.Pow(Math.E, -x));
            return sig;
        }
        double SigmoidTransferDerivative(double x)
        {
            double output;
            output = x * (1 - x);
            return output;
        }
        public double ErrorCalculation(double previousActivationValue)
        {
            double cost;
            //cost = Math.Pow((activationValue - desiredOutput),2);
            cost = previousActivationValue * SigmoidTransferDerivative(z) * 2 * (activationValue - desiredOutput);
            return (cost);

        }
    }
}
