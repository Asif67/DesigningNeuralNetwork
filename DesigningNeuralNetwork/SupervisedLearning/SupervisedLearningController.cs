using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DesigningNeuralNetwork.SupervisedLearning.OneInput;
using DesigningNeuralNetwork.SupervisedLearning.MultipleInput;



namespace DesigningNeuralNetwork.SupervisedLearning
{
    class SupervisedLearningController
    {
        public void SingleInputSingleHiddenLayerSingleNeuronSingleOutput()
        {
            DesigningNeuralNetwork.SupervisedLearning.OneInput.Input X = new DesigningNeuralNetwork.SupervisedLearning.OneInput.Input();
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
        public void SingleInputSingleHiddenLayerSingleNeuronMultipleOutput()
        {

            DesigningNeuralNetwork.SupervisedLearning.OneInput.Input X = new DesigningNeuralNetwork.SupervisedLearning.OneInput.Input();
            int numberOfOutputs;
            Console.WriteLine("Enter Number of Outputs(Max 1000):");
            numberOfOutputs = Convert.ToInt32(Console.ReadLine());
            OneInput.SingleHiddenLayer.SingleNeuron.Neuron N = new OneInput.SingleHiddenLayer.SingleNeuron.Neuron();
            OneInput.SingleHiddenLayer.SingleNeuron.MultipleOutput.Output[] O = new OneInput.SingleHiddenLayer.SingleNeuron.MultipleOutput.Output[numberOfOutputs];

            for (int l = 0; l < numberOfOutputs; l++)
            {
                O[l] = new OneInput.SingleHiddenLayer.SingleNeuron.MultipleOutput.Output();
            }
            //Initialization
            X.value = 1.0;
            N.weight_inputToHidden = 0.1;
            for (int l = 0; l < numberOfOutputs; l++)
            {
                N.weight_hiddenToOutputNo[l] = 0.1;//1000 Max; problem in dynamic array declare;
                O[l].target = 1.0;
            }
            //Initialization
            int numberOfEpoch;
            Console.WriteLine("Enter Number of Epoch");
            numberOfEpoch = Convert.ToInt32(Console.ReadLine());
            double total_weight_hiddenToOutput = 0;
            for (int i = 0; i < numberOfEpoch; i++)
            {
                N.Calculate(X.value);
                for (int k = 0; k < numberOfOutputs; k++)
                {
                    total_weight_hiddenToOutput += N.weight_hiddenToOutputNo[k];
                }
                for (int l = 0; l < numberOfOutputs; l++)
                {
                    O[l].Predict(N.value, N.weight_hiddenToOutputNo[l]);
                    O[l].ErrorCalculation();
                    Console.WriteLine("\nError[Epoch(" + i + "), Output(" + l + ")]= " + O[l].error +  " Output[Epoch(" + i +"), Output("+ l + ")]= " + O[l].guess + " Target = " + O[l].target);
                    Console.WriteLine("Input To Hidden Neuron Weight= " + N.weight_inputToHidden + " Hidden Neuron To Outputs Weights = " + N.weight_hiddenToOutputNo[l]);
                    N.Weight_Update_MultipleOutput(O[l].error,numberOfOutputs, total_weight_hiddenToOutput);
                }
            }
            Console.ReadKey();
        }
        public void SingleInputSingleHiddenLayerMultipleNeuronSingleOutput()
        {
            double totalOutput = 0;
            DesigningNeuralNetwork.SupervisedLearning.OneInput.Input X = new DesigningNeuralNetwork.SupervisedLearning.OneInput.Input();
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
        public void MultipleInputMultipleHiddenLayerMultipleNeuronMultipleOutput()
        {
            MultipleInput.Input X = new MultipleInput.Input();
            MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron N = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
            MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron N1 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
            MultipleInput.MultipleHiddenLayer.MultipleNeuron.MultipleOutput.Output O = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.MultipleOutput.Output();
            double previousActivationValue;
            double cost;

            X.input = 0.1;
            N.enteringCostWeight = 0.01;
            N.bias = 0.5;
            N1.enteringCostWeight = 0.02;
            N1.bias = 0.4;
            O.enteringCostWeight = 0.03;
            O.bias = 0.3;
            O.desiredOutput = 0.1;

            previousActivationValue = X.input;
            N.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N.activationValue);
            N1.ActivationvalueCalculation(N.activationValue);
            Console.WriteLine(N1.activationValue);
            O.ActivationvalueCalculation(N1.activationValue);
            Console.WriteLine(O.activationValue);
            cost = O.ErrorCalculation();
            Console.WriteLine(cost);
            Console.ReadKey();


        }
    }
}
