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
        //For multiple output
        public double[] weight_hiddenToOutputNo = new double[1000];
        public void Calculate(double input)
        {
            value = input * weight_inputToHidden;
        }
        public void Weight_Update(double error)
        {
            weight_inputToHidden = (weight_inputToHidden / weight_inputToHidden) * error;
            weight_hiddenToOutput = (weight_hiddenToOutput / weight_hiddenToOutput) * error;
        }
        public void Weight_Update_MultipleOutput(double error, int numberOfOutput, double totalWeight_hiddenToOutput)
        {
            //double[] weight_hiddenToOutputNo = new double[numberOfOutput];
            weight_inputToHidden = (weight_inputToHidden / weight_inputToHidden) * error;
            //For multiple output
            for (int i = 0; i < numberOfOutput; i++)
            {
                weight_hiddenToOutputNo[i] = (weight_hiddenToOutputNo[i] / totalWeight_hiddenToOutput) * error;
            }
        }
    }
}
