using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DesigningNeuralNetwork.SupervisedLearning.OneInput;


namespace DesigningNeuralNetwork.SupervisedLearning
{
    class SupervisedLearningController
    {
        public void SingleInputSingleHiddenLayerSingleNeuronSingleOutput()
        {
            Input X = new Input();
            OneInput.SingleHiddenLayer.SingleNeuron.Neuron N = new OneInput.SingleHiddenLayer.SingleNeuron.Neuron();
            OneInput.SingleHiddenLayer.SingleNeuron.OneOutput.Output O = new OneInput.SingleHiddenLayer.SingleNeuron.OneOutput.Output();
            //Initialization
            X.value = 1.0;
            N.weight_inputToHidden = 0.1;
            N.weight_hiddenToOutput = 0.1;
            O.target = 1.0;
            //Initialization
            for (int i = 0; i < 10; i++)
            {
                N.Calculate(X.value);
                O.Predict(N.value, N.weight_hiddenToOutput);
                O.ErrorCalculation();
                Console.WriteLine("Error[" + i + "]= " + O.error + " IH= " + N.weight_inputToHidden + " HO = " + N.weight_hiddenToOutput + " Target = " + O.target);
                N.Weight_Update(O.error);
            }
            Console.ReadKey();
        }
        public void SingleInputSingleHiddenLayerMultipleNeuronSingleOutput()
        {
            double totalOutput = 0;
            Input X = new Input();
            int numberOfHiddenNeurons;
            Console.WriteLine("Enter Number of Hidden Neurons");
            numberOfHiddenNeurons = Convert.ToInt32(Console.ReadLine());
            OneInput.SingleHiddenLayer.MultipleNeuron.Neuron[] N = new OneInput.SingleHiddenLayer.MultipleNeuron.Neuron[numberOfHiddenNeurons];
            
            for (int l = 0; l < numberOfHiddenNeurons; l++)
            {
                N[l] = new OneInput.SingleHiddenLayer.MultipleNeuron.Neuron();
            }
            
            OneInput.SingleHiddenLayer.MultipleNeuron.OneOutput.Output O = new OneInput.SingleHiddenLayer.MultipleNeuron.OneOutput.Output();
            //Initialization
            X.value = 1.0;
            for (int l = 0; l < numberOfHiddenNeurons; l++)
            {
                N[l].weight_inputToHidden = 0.1;
                N[l].weight_hiddenToOutput = 0.1;
            }
            
            double total_weight_inputToHidden=0;
            double total_weight_hiddenToOutput=0;
            
            O.target = 1.0;
            //Initialization
            /* i = number of epoch
             * j = number of hidden neurons
             * k = total weight calculate(imp)(do not modify)
             */
            int numberOfEpoch;
            Console.WriteLine("Enter Number of Epoch");
            numberOfEpoch = Convert.ToInt32(Console.ReadLine());

            for (int i = 0; i < numberOfEpoch; i++)
            {
                Console.WriteLine("Epoch ["+ i +"]:\n");
                for (int k = 0; k < numberOfHiddenNeurons; k++)
                {
                    total_weight_inputToHidden += N[k].weight_inputToHidden;
                    total_weight_hiddenToOutput += N[k].weight_hiddenToOutput;
                }
                for (int j = 0; j < numberOfHiddenNeurons; j++)
                {
                    N[j].Calculate(X.value);
                    O.Predict(N[j].value, N[j].weight_hiddenToOutput);
                    totalOutput += O.guess;
                    O.ErrorCalculation(totalOutput);
                    Console.WriteLine("Neuron [" + j + "]:");
                    Console.WriteLine("Error[" + i + "]= " + O.error + " Output[" + i + "]= " + O.guess + " Target = " + O.target);
                    Console.WriteLine("IH[" + j + "] = " + N[j].weight_inputToHidden + " HO[" + j + "] = " + N[j].weight_hiddenToOutput);
                    Console.WriteLine("Total Output = " + totalOutput + " Total IH = " + total_weight_inputToHidden + " Total OH = " + total_weight_hiddenToOutput + "\n");
                    
                    N[j].Weight_Update(O.error, total_weight_inputToHidden, total_weight_hiddenToOutput);
                }
            }
            Console.ReadKey();
        }
    }
}
