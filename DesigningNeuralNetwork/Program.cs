using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DesigningNeuralNetwork.SupervisedLearning.OneInput;
using DesigningNeuralNetwork.SupervisedLearning.OneInput.SingleHiddenLayer.SingleNeuron;
using DesigningNeuralNetwork.SupervisedLearning.OneInput.SingleHiddenLayer.SingleNeuron.OneOutput;

namespace DesigningNeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            Input X = new Input();
            Neuron N = new Neuron();
            Output O = new Output();
            //Initialization
            X.value = 1.0;
            N.weight_inputToHidden = 0.1;
            N.weight_hiddenToOutput = 0.1;
            O.target = 1.0;
            //Initialization
            for (int i=0;i<10;i++)
            {
                N.Calculate(X.value);
                O.Predict(N.value,N.weight_hiddenToOutput);
                O.ErrorCalculation();
                Console.WriteLine("Error[" + i + "]= " + O.error);
                N.Weight_Update(O.error);                                
            }
            Console.ReadKey();
        }
    }
}
