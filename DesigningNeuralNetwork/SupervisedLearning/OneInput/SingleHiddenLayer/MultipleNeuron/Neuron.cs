using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DesigningNeuralNetwork.SupervisedLearning.OneInput.SingleHiddenLayer.MultipleNeuron
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
        public void Weight_Update(double error, double total_weight_inputToHidden, double total_weight_hiddenToOutput)
        {
            weight_inputToHidden =  (weight_inputToHidden  / total_weight_inputToHidden) * error;
            weight_hiddenToOutput = (weight_hiddenToOutput / total_weight_hiddenToOutput) * error;
        }
    }
}
