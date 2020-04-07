using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DesigningNeuralNetwork.SupervisedLearning.OneInput.SingleHiddenLayer.SingleNeuron
{
    class Neuron
    {
        public double value = 0.0;
        public double weight_inputToHidden; 
        public double weight_hiddenToOutput;
        public void Calculate(double input)
        {
            value = input * weight_inputToHidden;
        }
        public void Weight_Update(double error)
        {
            weight_inputToHidden = (weight_inputToHidden / weight_inputToHidden) * error;
            weight_hiddenToOutput = (weight_hiddenToOutput / weight_hiddenToOutput) * error;
        }
    }
}
