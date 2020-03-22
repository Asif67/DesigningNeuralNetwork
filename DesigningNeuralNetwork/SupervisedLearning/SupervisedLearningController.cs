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
            //MultipleInput.Input X1 = new MultipleInput.Input();
            //MultipleInput.Input X2 = new MultipleInput.Input();
            //MultipleInput.Input X3 = new MultipleInput.Input();
            //MultipleInput.Input X4 = new MultipleInput.Input();
            //MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron N1HL1 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
            //MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron N2HL1 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
            //MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron N3HL1 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
            //MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron N1HL2 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
            //MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron N2HL2 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
            //MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron N3HL2 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
            //MultipleInput.MultipleHiddenLayer.MultipleNeuron.MultipleOutput.Output O1 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.MultipleOutput.Output();
            //MultipleInput.MultipleHiddenLayer.MultipleNeuron.MultipleOutput.Output O2 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.MultipleOutput.Output();
            
            //double previousActivationValue;
            //double cost;

            //X1.input = 0.1;
            //X2.input = 0.1;
            //X3.input = 0.1;
            //X4.input = 0.1;

            //N1HL1.enteringCostWeight = 0.01;
            //N1HL1.bias = 0.5;
            
            //N2HL1.enteringCostWeight = 0.01;
            //N2HL1.bias = 0.5;
            
            //N3HL1.enteringCostWeight = 0.01;
            //N3HL1.bias = 0.5;
            
            //N1HL2.enteringCostWeight = 0.02;
            //N1HL2.bias = 0.4;
            
            //N2HL2.enteringCostWeight = 0.02;
            //N2HL2.bias = 0.4;
            
            //N3HL2.enteringCostWeight = 0.02;
            //N3HL2.bias = 0.4;
            
            //O1.enteringCostWeight = 0.03;
            //O1.bias = 0.3;
            //O1.desiredOutput = 0.1;
            
            //O2.enteringCostWeight = 0.03;
            //O2.bias = 0.3;
            //O2.desiredOutput = 0.1;



            //void N1HL1F()
            //{
            //    previousActivationValue = X1.input;
            //    N1HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X1.input + " * " + N1HL1.enteringCostWeight + " = " + N1HL1.weightedSum);

            //    previousActivationValue = X2.input;
            //    N1HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X2.input + " * " + N1HL1.enteringCostWeight + " = " + N1HL1.weightedSum);

            //    previousActivationValue = X3.input;
            //    N1HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X3.input + " * " + N1HL1.enteringCostWeight + " = " + N1HL1.weightedSum);

            //    previousActivationValue = X4.input;
            //    N1HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X4.input + " * " + N1HL1.enteringCostWeight + " = " + N1HL1.weightedSum);

            //    N1HL1.SigmoidActivationFunction();
            //    Console.WriteLine(X1.input + " * " + N1HL1.enteringCostWeight + " + " + X2.input + " * " + N1HL1.enteringCostWeight + " + " + X3.input + " * " + N1HL1.enteringCostWeight + " + " + X4.input + " * " + N1HL1.enteringCostWeight + " + " + N1HL1.bias + " = " + N1HL1.x);
            //    Console.WriteLine("1/(1+e^-" + N1HL1.x + ") = " + N1HL1.activationValue);
            //}

            //void N2HL1F()
            //{
            //    previousActivationValue = X1.input;
            //    N2HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X1.input + " * " + N2HL1.enteringCostWeight + " = " + N2HL1.weightedSum);

            //    previousActivationValue = X2.input;
            //    N2HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X2.input + " * " + N2HL1.enteringCostWeight + " = " + N2HL1.weightedSum);

            //    previousActivationValue = X3.input;
            //    N2HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X3.input + " * " + N2HL1.enteringCostWeight + " = " + N2HL1.weightedSum);

            //    previousActivationValue = X4.input;
            //    N2HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X4.input + " * " + N2HL1.enteringCostWeight + " = " + N2HL1.weightedSum);

            //    N2HL1.SigmoidActivationFunction();
            //    Console.WriteLine(X1.input + " * " + N2HL1.enteringCostWeight + " + " + X2.input + " * " + N2HL1.enteringCostWeight + " + " + X3.input + " * " + N2HL1.enteringCostWeight + " + " + X4.input + " * " + N2HL1.enteringCostWeight + " + " + N2HL1.bias + " = " + N2HL1.x);
            //    Console.WriteLine("1/(1+e^-" + N2HL1.x + ") = " + N2HL1.activationValue);

            //}

            //void N3HL1F()
            //{
            //    previousActivationValue = X1.input;
            //    N3HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X1.input + " * " + N3HL1.enteringCostWeight + " = " + N3HL1.weightedSum);

            //    previousActivationValue = X2.input;
            //    N2HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X2.input + " * " + N3HL1.enteringCostWeight + " = " + N3HL1.weightedSum);

            //    previousActivationValue = X3.input;
            //    N2HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X3.input + " * " + N3HL1.enteringCostWeight + " = " + N3HL1.weightedSum);

            //    previousActivationValue = X4.input;
            //    N2HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X4.input + " * " + N3HL1.enteringCostWeight + " = " + N3HL1.weightedSum);

            //    N2HL1.SigmoidActivationFunction();
            //    Console.WriteLine(X1.input + " * " + N3HL1.enteringCostWeight + " + " + X2.input + " * " + N3HL1.enteringCostWeight + " + " + X3.input + " * " + N3HL1.enteringCostWeight + " + " + X4.input + " * " + N3HL1.enteringCostWeight + " + " + N3HL1.bias + " = " + N3HL1.x);
            //    Console.WriteLine("1/(1+e^-" + N3HL1.x + ") = " + N3HL1.activationValue);

            //}

            //void N1HL2F()
            //{
            //    previousActivationValue = X1.input;
            //    N1HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X1.input + " * " + N1HL1.enteringCostWeight + " = " + N1HL1.weightedSum);

            //    previousActivationValue = X2.input;
            //    N1HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X2.input + " * " + N1HL1.enteringCostWeight + " = " + N1HL1.weightedSum);

            //    previousActivationValue = X3.input;
            //    N1HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X3.input + " * " + N1HL1.enteringCostWeight + " = " + N1HL1.weightedSum);

            //    previousActivationValue = X4.input;
            //    N1HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X4.input + " * " + N1HL1.enteringCostWeight + " = " + N1HL1.weightedSum);

            //    N1HL1.SigmoidActivationFunction();
            //    Console.WriteLine(X1.input + " * " + N1HL1.enteringCostWeight + " + " + X2.input + " * " + N1HL1.enteringCostWeight + " + " + X3.input + " * " + N1HL1.enteringCostWeight + " + " + X4.input + " * " + N1HL1.enteringCostWeight + " + " + N1HL1.bias + " = " + N1HL1.x);
            //    Console.WriteLine("1/(1+e^-" + N1HL1.x + ") = " + N1HL1.activationValue);
            //}

            //void N2HL2F()
            //{
            //    previousActivationValue = X1.input;
            //    N2HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X1.input + " * " + N2HL1.enteringCostWeight + " = " + N2HL1.weightedSum);

            //    previousActivationValue = X2.input;
            //    N2HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X2.input + " * " + N2HL1.enteringCostWeight + " = " + N2HL1.weightedSum);

            //    previousActivationValue = X3.input;
            //    N2HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X3.input + " * " + N2HL1.enteringCostWeight + " = " + N2HL1.weightedSum);

            //    previousActivationValue = X4.input;
            //    N2HL1.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(X4.input + " * " + N2HL1.enteringCostWeight + " = " + N2HL1.weightedSum);

            //    N2HL1.SigmoidActivationFunction();
            //    Console.WriteLine(X1.input + " * " + N2HL1.enteringCostWeight + " + " + X2.input + " * " + N2HL1.enteringCostWeight + " + " + X3.input + " * " + N2HL1.enteringCostWeight + " + " + X4.input + " * " + N2HL1.enteringCostWeight + " + " + N2HL1.bias + " = " + N2HL1.x);
            //    Console.WriteLine("1/(1+e^-" + N2HL1.x + ") = " + N2HL1.activationValue);

            //}

            //void N3HL2F()
            //{
            //    previousActivationValue = N1HL1.activationValue;
            //    N3HL2.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(N1HL1.activationValue + " * " + N3HL2.enteringCostWeight + " = " + N3HL2.weightedSum);

            //    previousActivationValue = N2HL1.activationValue;
            //    N3HL2.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(N2HL1.activationValue + " * " + N3HL2.enteringCostWeight + " = " + N3HL2.weightedSum);

            //    previousActivationValue = N3HL1.activationValue;
            //    N3HL2.ActivationvalueCalculation(previousActivationValue);
            //    Console.WriteLine(N3HL1.activationValue + " * " + N3HL2.enteringCostWeight + " = " + N3HL2.weightedSum);

            //    N3HL2.SigmoidActivationFunction();
            //    Console.WriteLine(N3HL1.activationValue + " * " + N3HL2.enteringCostWeight + " + " + N3HL1.activationValue + " * " + N3HL2.enteringCostWeight + " + " + N3HL1.activationValue + " * " + N3HL2.enteringCostWeight + " + " + N3HL2.bias + " = " + N3HL2.x);
            //    Console.WriteLine("1/(1+e^-" + N3HL2.x + ") = " + N3HL2.activationValue);

            //}

            //Console.WriteLine("Input to Hidden Layer 1");
            //void HiddenLayer1()
            //{
            //    Console.WriteLine("a(1,1-3) to a(1,1):");
            //    N1HL1F();
            //    Console.WriteLine("\na(1,1-3) to a(1,2):");
            //    N2HL1F();
            //    Console.WriteLine("\na(1,1-3) to a(1,3):");
            //    N3HL1F();
            //}
            
            //Console.WriteLine("\nHidden Layer 1 to Hidden Layer 2");
            //void HiddenLayer2()
            //{
            //    Console.WriteLine("a(1,1-2) to a(2,1):");
            //    N1HL2F();
            //    Console.WriteLine("\na(1,1-2) to a(2,2):");
            //    N2HL2F();
            //    Console.WriteLine("\na(1,1-2) to a(2,3):");
            //    N3HL2F();
            //}
            
            
            ////N.ActivationvalueCalculation(previousActivationValue);
            ////Console.WriteLine(N.activationValue);
            ////N.ActivationvalueCalculation(previousActivationValue);
            ////Console.WriteLine(N.activationValue);
            ////N1.ActivationvalueCalculation(N.activationValue);
            ////Console.WriteLine(N1.activationValue);
            ////O.ActivationvalueCalculation(N1.activationValue);
            ////Console.WriteLine(O.activationValue);
            //Console.ReadKey();


        }
        
    }
}
