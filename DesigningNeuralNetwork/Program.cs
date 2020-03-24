using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DesigningNeuralNetwork.SupervisedLearning;

namespace DesigningNeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            SupervisedLearningController S = new SupervisedLearningController();
            //S.SingleInputSingleHiddenLayerSingleNeuronSingleOutput();
            //S.SingleInputSingleHiddenLayerMultipleNeuronSingleOutput();
            //S.SingleInputSingleHiddenLayerSingleNeuronMultipleOutput();
            S.MultipleInputMultipleHiddenLayerMultipleNeuronMultipleOutput();
            //test t = new test();
            //t.Initialize();
            //t.HiddenLayer1();
            //t.HiddenLayer2();
            //t.Output();
            //t.MultipleInputMultipleHiddenLayerMultipleNeuronMultipleOutput();
        }
        
    }
}
