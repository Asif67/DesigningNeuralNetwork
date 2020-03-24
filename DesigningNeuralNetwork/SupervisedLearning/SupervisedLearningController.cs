using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DesigningNeuralNetwork.SupervisedLearning.OneInput;
using DesigningNeuralNetwork.SupervisedLearning.MultipleInput;
using System.Windows.Media;

namespace DesigningNeuralNetwork.SupervisedLearning
{
    class SupervisedLearningController
    {
        //variables needed for 1 Sample, 4 Inputs, 2 Hidden Layers(3 Hidden Neurons each), 2 Outputs
        //static int numberOfInputNeurons = 4;
        //static int columnMatrixDef = 1;
        //static int HL1NumberofNeurons = 3;
        //static int HL2NumberofNeurons = 3;
        //static int numberOfOutputNeurons = 2;

        static int numberOfInputNeurons = 784;
        static int columnMatrixDef = 1;
        static int HL1NumberofNeurons = 16;
        static int HL2NumberofNeurons = 16;
        static int numberOfOutputNeurons = 10;


        double[,] inputToHiddenLayer1WeightMatrix = new double[HL1NumberofNeurons, numberOfInputNeurons];//3 rows 4 coloums
        double[,] hiddenLayer1ToHiddenLayer2WeightMatrix = new double[HL2NumberofNeurons, HL1NumberofNeurons];//3 rows 3 coloums
        double[,] hiddenLayer2ToOutputWeightMatrix = new double[HL2NumberofNeurons, numberOfOutputNeurons];//3 rows 2 coloums

        double[,] inputActivation = new double[numberOfInputNeurons, columnMatrixDef];//4 rows 1 coloums
        double[,] hiddenLayer1Activation = new double[HL1NumberofNeurons, columnMatrixDef];//3 rows 1 coloums
        double[,] hiddenLayer2Activation = new double[HL2NumberofNeurons, columnMatrixDef];//3 rows 1 coloums
        double[,] outputActivation = new double[numberOfOutputNeurons, columnMatrixDef];//2 rows 1 coloums

        double[,] hiddenLayer1Bias = new double[HL1NumberofNeurons, columnMatrixDef];//3 rows 1 coloums
        double[,] hiddenLayer2Bias = new double[HL2NumberofNeurons, columnMatrixDef];//3 rows 1 coloums
        double[,] outputBias = new double[numberOfOutputNeurons, columnMatrixDef];//2 rows 1 coloums

        double[,] z1 = new double[HL1NumberofNeurons, columnMatrixDef];//3 rows 1 coloums
        double[,] z2 = new double[HL2NumberofNeurons, columnMatrixDef];//3 rows 1 coloums
        double[,] z3 = new double[numberOfOutputNeurons, columnMatrixDef];//2 rows 1 coloums
        double[,] desiredOutput = new double[numberOfOutputNeurons, columnMatrixDef];//2 rows 1 coloums
        double[,] cost = new double[numberOfOutputNeurons, columnMatrixDef];//2 rows 1 coloums
        double totalCost;
        //variables needed for 1 Sample, 4 Inputs, 2 Hidden Layers(3 Hidden Neurons each), 2 Outputs
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
            Initialize();
            HiddenLayers();
            Output();
            CostCalculation();
        }
        void Initialize()
        {
            WeightMatrixInitialize();
            ActivationInitialize();
            BiasInitialize();
            DesiredOutputInitialize();
        }
        void HiddenLayers()
        {
            HiddenLayer1();
            HiddenLayer2();
        }
        void HiddenLayer1()
        {

            Console.WriteLine("\nHidden Layer 1:");
            z1 = MM(inputToHiddenLayer1WeightMatrix, HL1NumberofNeurons, numberOfInputNeurons, inputActivation, numberOfInputNeurons, columnMatrixDef);
            z1 = MA(z1, HL1NumberofNeurons, columnMatrixDef, hiddenLayer1Bias, HL1NumberofNeurons, columnMatrixDef);
            Console.WriteLine("\nAfter SigmoidActivation Function application:");
            for (int i = 0; i < HL1NumberofNeurons; i++)
            {
                z1[i, 0] = SigmoidActivationFunction(z1[i, 0]);
                Console.WriteLine(z1[i, 0]);
            }
            Console.ReadKey();
        }
        void HiddenLayer2()
        {
            Console.WriteLine("\nHidden Layer 2:");
            z2 = MM(hiddenLayer1ToHiddenLayer2WeightMatrix, HL2NumberofNeurons, HL1NumberofNeurons, hiddenLayer1Activation, HL2NumberofNeurons, columnMatrixDef);
            z2 = MA(z2, HL2NumberofNeurons, columnMatrixDef, hiddenLayer2Bias, HL2NumberofNeurons, columnMatrixDef);
            Console.WriteLine("\nAfter SigmoidActivation Function application:");
            for (int i = 0; i < HL2NumberofNeurons; i++)
            {
                z2[i, 0] = SigmoidActivationFunction(z2[i, 0]);
                Console.WriteLine(z2[i, 0]);
            }
            Console.ReadKey();
        }
        void Output()
        {
            Console.WriteLine("\nOutput:");
            z3 = MM(hiddenLayer2ToOutputWeightMatrix, HL2NumberofNeurons, numberOfOutputNeurons, outputActivation, numberOfOutputNeurons, columnMatrixDef);
            z3 = MA(z3, numberOfOutputNeurons, columnMatrixDef, outputBias, numberOfOutputNeurons, columnMatrixDef);
            Console.WriteLine("\nAfter SigmoidActivation Function application:");
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                z3[i, 0] = SigmoidActivationFunction(z3[i, 0]);
                Console.WriteLine(z3[i, 0]);
            }
            Console.ReadKey();
        }
        double SigmoidActivationFunction(double x)
        {
            double sig = 1 / (1 + Math.Pow(Math.E, -x));
            return sig;
        }
        void WeightMatrixInitialize()
        {
            
            //Initialize Weight Matrix
            for (int i = 0; i < HL1NumberofNeurons; i++)
            {
                for (int j = 0; j < numberOfInputNeurons; j++)
                {
                    Random X1 = new Random();
                    inputToHiddenLayer1WeightMatrix[i, j] = X1.NextDouble();
                    //inputToHiddenLayer1WeightMatrix[i, j] = 0.01;
                    Console.WriteLine("IH1WM(" + i + "," + j + ") = " + inputToHiddenLayer1WeightMatrix[i, j]);
                }
                Console.WriteLine("\n");
            }
            for (int i = 0; i < HL2NumberofNeurons; i++)
            {
                for (int j = 0; j < HL1NumberofNeurons; j++)
                {
                    Random X2 = new Random();
                    hiddenLayer1ToHiddenLayer2WeightMatrix[i, j] = X2.NextDouble();
                    //hiddenLayer1ToHiddenLayer2WeightMatrix[i, j] = 0.02;
                    Console.WriteLine("H1H2WM(" + i + "," + j + ") = " + hiddenLayer1ToHiddenLayer2WeightMatrix[i, j]);
                }
                Console.WriteLine("\n");
            }
            for (int i = 0; i < HL2NumberofNeurons; i++)
            {
                for (int j = 0; j < numberOfOutputNeurons; j++)
                {
                    Random X3 = new Random();
                    hiddenLayer2ToOutputWeightMatrix[i, j] = X3.NextDouble();
                    //hiddenLayer2ToOutputWeightMatrix[i, j] = 0.03;
                    Console.WriteLine("H2OWM(" + i + "," + j + ") = " + hiddenLayer2ToOutputWeightMatrix[i, j]);
                }
                Console.WriteLine("\n");
            }
            //Initialize Weight Matrix
        }
        void ActivationInitialize()
        {
            
            //Initialize Activation Matrix
            for (int i = 0; i < numberOfInputNeurons; i++)
            {
                Random X1 = new Random();
                inputActivation[i, 0] = X1.NextDouble();
                //inputActivation[i, 0] = 0.1;
                Console.WriteLine("IA(" + 0 + "," + i + ") = " + inputActivation[i,0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < HL1NumberofNeurons; i++)
            {
                Random X2 = new Random();
                hiddenLayer1Activation[i, 0] = X2.NextDouble();
                //hiddenLayer1Activation[i, 0] = 0.2;
                Console.WriteLine("H1A(" + 0 + "," + i + ") = " + hiddenLayer1Activation[i,0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < HL2NumberofNeurons; i++)
            {
                Random X3 = new Random();
                hiddenLayer2Activation[i,0] = X3.NextDouble();
                //hiddenLayer2Activation[i, 0] = 0.3;
                Console.WriteLine("H2A(" + 0 + "," + i + ") = " + hiddenLayer2Activation[i,0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                Random X4 = new Random();
                outputActivation[i,0] = X4.NextDouble();
                //outputActivation[i, 0] = 0.4;
                Console.WriteLine("OA(" + 0 + "," + i + ") = " + outputActivation[i, 0]);
            }
            Console.WriteLine("\n");
            //Initialize Activation Matrix
        }
        void BiasInitialize()
        {
            
            //Initialize bias Matrix
            for (int i = 0; i < HL1NumberofNeurons; i++)
            {
                Random X1 = new Random();
                hiddenLayer1Bias[i, 0] = X1.NextDouble();
                //hiddenLayer1Bias[i, 0] = 0.001;
                Console.WriteLine("H1B(" + 0 + "," + i + ") = " + hiddenLayer1Bias[i,0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < HL2NumberofNeurons; i++)
            {
                Random X2 = new Random();
                hiddenLayer2Bias[i, 0] = X2.NextDouble();
                //hiddenLayer2Bias[i, 0] = 0.002;
                Console.WriteLine("H2B(" + 0 + "," + i + ") = " + hiddenLayer2Bias[i, 0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                Random X3 = new Random();
                outputBias[i, 0] = X3.NextDouble();
                //outputBias[i, 0] = 0.003;
                Console.WriteLine("OB(" + 0 + "," + i + ") = " + outputBias[i, 0]);
            }
            Console.WriteLine("\n");
            //Initialize bias Matrix
        }
        void DesiredOutputInitialize()
        {
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                desiredOutput[i, 0] = 1;
                Console.WriteLine(desiredOutput[i, 0]);
            }
            Console.ReadKey();
        }
        void CostCalculation()
        {
            Console.WriteLine("\nCost(desiredoutput - predicted output):");
            cost = MS(z3, numberOfOutputNeurons, columnMatrixDef, desiredOutput, numberOfOutputNeurons, columnMatrixDef);
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                Console.WriteLine(cost[i, 0]);
            }
            Console.WriteLine("\nCost(desiredoutput - predicted output)^2:");
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                cost[i, 0] = Math.Pow((cost[i,0]), 2);
                totalCost += cost[i, 0];
                Console.WriteLine(cost[i, 0]);
            }
            Console.WriteLine(totalCost);
            /* 
             * aL = activation value of the output neuron
             * aL-1 = activation value of the previous neuron
             * wL = weight value of the output neuron
             * bL = bias value of the output neuron
             * c0 = (aL-desiredOutput)^2
             * aL = SigmoidActivationFunction(zL)
             * zL = wL * aL-1 + bL
             * dc0/dwL = (dzL/dwL) * (daL/dzL) * (dc0/daL) //Chain rule
             * dzL/dwL = aL-1
             * daL/dzL = FirstOrderDerivationOfSigmoidFunction(zL)
             * dc0/daL = 2 * (aL - desiredOutput)
             * dc0/dwL = (aL-1) * (FirstOrderDerivationOfSigmoidFunction(zL)) * (2 * (aL - desiredOutput)) //Simplified Chain rule for source code use
             */
            Console.WriteLine("\nCost(Simplified Chain rule applied):");
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                cost[i, 0] = Math.Pow((cost[i, 0]), 2);
                Console.WriteLine(cost[i, 0]);
            }
            Console.ReadKey();
        }
        double[,] MM(double[,] a, int m,int n, double[,] b, int p,int q)
        {
            int i, j;
            Console.WriteLine("Matrix a:");
            for (i = 0; i < m; i++)
            {
                for (j = 0; j < n; j++)
                {
                    Console.Write(a[i, j] + " ");
                }
                Console.WriteLine();
            }
            Console.WriteLine("Matrix b:");
            for (i = 0; i < p; i++)
            {
                for (j = 0; j < q; j++)
                {
                    Console.Write(b[i, j] + " ");
                }
                Console.WriteLine();
            }
            if (n != p)
                {
                    Console.WriteLine("Matrix multiplication not possible");
                return (a);
                }
                else
                {
                    double[,] c = new double[m, n];
                    for (i = 0; i < m; i++)
                    {
                        for (j = 0; j < q; j++)
                        {
                            c[i, j] = 0;
                            for (int k = 0; k < n; k++)
                            {
                                c[i, j] += a[i, k] * b[k, j];
                            }
                        }
                    }
                Console.WriteLine("The product of the two matrices is :");
                for (i = 0; i < m; i++)
                {
                    for (j = 0; j < q; j++)
                    {
                        Console.Write(c[i, j] + "\t");
                    }
                    Console.WriteLine();
                }
                return (c);
                }          
        }
        double[,] MA(double[,] arr1, int m, int n, double[,] arr2,int p, int q)
        {
            int i, j;
            double[,] arr3 = new double[m, q];
            Console.Write("\nFirst matrix is:\n");
            for (i = 0; i < m; i++)
            {
                Console.Write("\n");
                for (j = 0; j < q; j++)
                    Console.Write("{0}\t", arr1[i, j]);
            }
            Console.Write("\nSecond matrix is:\n");
            for (i = 0; i < m; i++)
            {
                Console.Write("\n");
                for (j = 0; j < q; j++)
                    Console.Write("{0}\t", arr2[i, j]);
            }
            for (i = 0; i < m; i++)
                for (j = 0; j < q; j++)
                    arr3[i, j] = arr1[i, j] + arr2[i, j];
            Console.Write("\nAdding two matrices: \n");
            for (i = 0; i < m; i++)
            {
                Console.Write("\n");
                for (j = 0; j < q; j++)
                    Console.Write("{0}\t", arr3[i, j]);
            }
            Console.Write("\n\n");
            return (arr3);
        }
        double[,] MS(double[,] arr1, int m, int n, double[,] arr2, int p, int q)
        {
            int i, j;
            double[,] arr3 = new double[m, q];
            //Console.Write("\nFirst matrix is:\n");
            //for (i = 0; i < m; i++)
            //{
            //    Console.Write("\n");
            //    for (j = 0; j < q; j++)
            //        Console.Write("{0}\t", arr1[i, j]);
            //}
            //Console.Write("\nSecond matrix is:\n");
            //for (i = 0; i < m; i++)
            //{
            //    Console.Write("\n");
            //    for (j = 0; j < q; j++)
            //        Console.Write("{0}\t", arr2[i, j]);
            //}
            for (i = 0; i < m; i++)
                for (j = 0; j < q; j++)
                    arr3[i, j] = arr1[i, j] - arr2[i, j];
            //Console.Write("\nAdding two matrices: \n");
            //for (i = 0; i < m; i++)
            //{
            //    Console.Write("\n");
            //    for (j = 0; j < q; j++)
            //        Console.Write("{0}\t", arr3[i, j]);
            //}
            //Console.Write("\n\n");
            return (arr3);
        }
    }
}
