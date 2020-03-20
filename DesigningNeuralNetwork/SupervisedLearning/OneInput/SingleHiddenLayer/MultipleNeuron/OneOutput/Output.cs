using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DesigningNeuralNetwork.SupervisedLearning.OneInput.SingleHiddenLayer.MultipleNeuron.OneOutput
{
    class Output
    {
        public double target;
        public double error = 0.0;
        public double guess = 0.0;
        public void Predict(double neuronValue, double weight_hiddenToOutput)
        {
            guess = neuronValue * weight_hiddenToOutput;
        }
        public void ErrorCalculation(double totalGuess)
        {
            error = target - totalGuess;
        }
    }
}
