using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DesigningNeuralNetwork.SupervisedLearning.OneInput;
using DesigningNeuralNetwork.SupervisedLearning.MultipleInput;

namespace DesigningNeuralNetwork.SupervisedLearning
{
    class test
    {
        //variables needed for 1 Sample, 1 Inputs, 2 Hidden Layers(1 Hidden Neurons each), 1 Outputs
        static int row=1, column=1;
        static int numberOfNeuronsHiddenLayer1=3, columnInputLayer=784;
        double[,] inputToHiddenLayer1WeightMatrix = new double[numberOfNeuronsHiddenLayer1, columnInputLayer];//3 rows 4 coloums
        double[,] hiddenLayer1ToHiddenLayer2WeightMatrix = new double[row, column];//3 rows 3 coloums
        double[,] hiddenLayer2ToOutputWeightMatrix = new double[row, column];//3 rows 2 coloums

        double[,] inputActivation = new double[columnInputLayer, 1];//4 rows 1 coloums
        double[,] hiddenLayer1Activation = new double[row, column];//3 rows 1 coloums
        double[,] hiddenLayer2Activation = new double[row, column];//3 rows 1 coloums
        double[,] outputActivation = new double[row, column];//2 rows 1 coloums

        double[,] hiddenLayer1Bias = new double[row, column];//3 rows 1 coloums
        double[,] hiddenLayer2Bias = new double[row, column];//3 rows 1 coloums
        double[,] outputBias = new double[row, column];//2 rows 1 coloums

        double[,] z1 = new double[row, column];//3 rows 1 coloums
        double[,] z2 = new double[row, column];//3 rows 1 coloums
        double[,] z3 = new double[row, column];//2 rows 1 coloums
        double[,] desiredOutput = new double[row, column];//2 rows 1 coloums
        double[,] cost = new double[row, column];//2 rows 1 coloums
        double totalCost;
        //variables needed for 1 Sample, 1 Inputs, 2 Hidden Layers(1 Hidden Neurons each), 1 Outputs

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
            z1 = MM(inputToHiddenLayer1WeightMatrix, row, column, inputActivation, row, column);
            z1 = MA(z1, row, column, hiddenLayer1Bias, row, column);
            for (int i = 0; i < row; i++)
            {
                z1[i, 0] = SigmoidActivationFunction(z1[i, 0]);
                Console.WriteLine(z1[i, 0]);
            }
            Console.ReadKey();
        }
        void HiddenLayer2()
        {
            Console.WriteLine("\nHidden Layer 2:");
            z2 = MM(hiddenLayer1ToHiddenLayer2WeightMatrix, row, column, hiddenLayer1Activation, row,column);
            z2 = MA(z2, row, column, hiddenLayer2Bias, row, column);
            for (int i = 0; i < row; i++)
            {
                z2[i, 0] = SigmoidActivationFunction(z2[i, 0]);
                Console.WriteLine(z2[i, 0]);
            }
            Console.ReadKey();
        }
        void Output()
        {
            Console.WriteLine("\nOutput:");
            z3 = MM(hiddenLayer2ToOutputWeightMatrix, row, column, outputActivation, column,row);
            z3 = MA(z3, row, column, outputBias, row, column);
            for (int i = 0; i < row; i++)
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
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    Random X1 = new Random();
                    //inputToHiddenLayer1WeightMatrix[i, j] = X1.NextDouble();
                    inputToHiddenLayer1WeightMatrix[i, j] = 0.01;
                    Console.WriteLine("IH1WM(" + i + "," + j + ") = " + inputToHiddenLayer1WeightMatrix[i, j]);
                }
                Console.WriteLine("\n");
            }
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    Random X2 = new Random();
                    //hiddenLayer1ToHiddenLayer2WeightMatrix[i, j] = X2.NextDouble();
                    hiddenLayer1ToHiddenLayer2WeightMatrix[i, j] = 0.02;
                    Console.WriteLine("H1H2WM(" + i + "," + j + ") = " + hiddenLayer1ToHiddenLayer2WeightMatrix[i, j]);
                }
                Console.WriteLine("\n");
            }
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    Random X3 = new Random();
                    //hiddenLayer2ToOutputWeightMatrix[i, j] = X3.NextDouble();
                    hiddenLayer2ToOutputWeightMatrix[i, j] = 0.03;
                    Console.WriteLine("H2OWM(" + i + "," + j + ") = " + hiddenLayer2ToOutputWeightMatrix[i, j]);
                }
                Console.WriteLine("\n");
            }
            //Initialize Weight Matrix
        }
        void ActivationInitialize()
        {

            //Initialize Activation Matrix
            for (int i = 0; i < row; i++)
            {
                Random X1 = new Random();
                //inputActivation[0, i] = X1.NextDouble();
                inputActivation[i, 0] = 0.1;
                Console.WriteLine("IA(" + 0 + "," + i + ") = " + inputActivation[i, 0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < row; i++)
            {
                Random X2 = new Random();
                //hiddenLayer1Activation[0, i] = X2.NextDouble();
                hiddenLayer1Activation[i, 0] = 0.2;
                Console.WriteLine("H1A(" + 0 + "," + i + ") = " + hiddenLayer1Activation[i, 0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < row; i++)
            {
                Random X3 = new Random();
                //hiddenLayer2Activation[0, i] = X3.NextDouble();
                hiddenLayer2Activation[i, 0] = 0.3;
                Console.WriteLine("H2A(" + 0 + "," + i + ") = " + hiddenLayer2Activation[i, 0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < row; i++)
            {
                Random X4 = new Random();
                //outputActivation[0, i] = X4.NextDouble();
                outputActivation[i, 0] = 0.4;
                Console.WriteLine("OA(" + 0 + "," + i + ") = " + outputActivation[i, 0]);
            }
            Console.WriteLine("\n");
            //Initialize Activation Matrix
        }
        void BiasInitialize()
        {

            //Initialize bias Matrix
            for (int i = 0; i < row; i++)
            {
                Random X1 = new Random();
                //hiddenLayer1Bias[i, 0] = X1.NextDouble();
                hiddenLayer1Bias[i, 0] = 0.001;
                Console.WriteLine("H1B(" + 0 + "," + i + ") = " + hiddenLayer1Bias[i, 0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < row; i++)
            {
                Random X2 = new Random();
                //hiddenLayer2Bias[i, 0] = X2.NextDouble();
                hiddenLayer2Bias[i, 0] = 0.002;
                Console.WriteLine("H2B(" + 0 + "," + i + ") = " + hiddenLayer2Bias[i, 0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < row; i++)
            {
                Random X3 = new Random();
                //outputBias[i, 0] = X3.NextDouble();
                outputBias[i, 0] = 0.003;
                Console.WriteLine("OB(" + 0 + "," + i + ") = " + outputBias[i, 0]);
            }
            Console.WriteLine("\n");
            //Initialize bias Matrix
        }
        void DesiredOutputInitialize()
        {
            for (int i = 0; i < row; i++)
            {
                desiredOutput[i, 0] = 1;
                Console.WriteLine(desiredOutput[i, 0]);
            }
            Console.ReadKey();
        }
        void CostCalculation()
        {
            Console.WriteLine("\nCost(desiredoutput - predicted output):");
            cost = MS(z3, row, column, desiredOutput, row, column);
            for (int i = 0; i < row; i++)
            {
                Console.WriteLine(cost[i, 0]);
            }
            Console.WriteLine("\nCost(desiredoutput - predicted output)^2:");
            for (int i = 0; i < row; i++)
            {
                cost[i, 0] = Math.Pow((cost[i, 0]), 2);
                totalCost += cost[i, 0];
                Console.WriteLine(cost[i, 0]);
            }
            Console.WriteLine("c0: ");
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
            for (int i = 0; i < row; i++)
            {
                //cost[i, 0] = Math.Pow((cost[i, 0]), 2);
                //Console.WriteLine(cost[i, 0]);
            }
            Console.ReadKey();
        }
        double[,] MM(double[,] a, int m, int n, double[,] b, int p, int q)
        {
            int i, j;
            ////Console.WriteLine("Matrix a:");
            //for (i = 0; i < m; i++)
            //{
            //    for (j = 0; j < n; j++)
            //    {
            //        //Console.Write(a[i, j] + " ");
            //    }
            //    //Console.WriteLine();
            //}
            ////Console.WriteLine("Matrix b:");
            //for (i = 0; i < p; i++)
            //{
            //    for (j = 0; j < q; j++)
            //    {
            //        Console.Write(b[i, j] + " ");
            //    }
            //    Console.WriteLine();
            //}
            if (n != p)
            {
                Console.WriteLine("Matrix multiplication not possible");
                return (a);
            }
            else
            {
                double[,] c = new double[m, q];
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
                //Console.WriteLine("The product of the two matrices is :");
                //for (i = 0; i < m; i++)
                //{
                //    for (j = 0; j < q; j++)
                //    {
                //        Console.Write(c[i, j] + "\t");
                //    }
                //    Console.WriteLine();
                //}
                return (c);
            }
        }
        double[,] MA(double[,] arr1, int m, int n, double[,] arr2, int p, int q)
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
                    arr3[i, j] = arr1[i, j] + arr2[i, j];
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
        //MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron O1 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
        //MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron O2 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
        ////MultipleInput.MultipleHiddenLayer.MultipleNeuron.MultipleOutput.Output O1 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.MultipleOutput.Output();
        ////MultipleInput.MultipleHiddenLayer.MultipleNeuron.MultipleOutput.Output O2 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.MultipleOutput.Output();

        //double previousActivationValue;
        //double cost;
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
        //    N3HL1.ActivationvalueCalculation(previousActivationValue);
        //    Console.WriteLine(X2.input + " * " + N3HL1.enteringCostWeight + " = " + N3HL1.weightedSum);

        //    previousActivationValue = X3.input;
        //    N3HL1.ActivationvalueCalculation(previousActivationValue);
        //    Console.WriteLine(X3.input + " * " + N3HL1.enteringCostWeight + " = " + N3HL1.weightedSum);

        //    previousActivationValue = X4.input;
        //    N3HL1.ActivationvalueCalculation(previousActivationValue);
        //    Console.WriteLine(X4.input + " * " + N3HL1.enteringCostWeight + " = " + N3HL1.weightedSum);

        //    N3HL1.SigmoidActivationFunction();
        //    Console.WriteLine(X1.input + " * " + N3HL1.enteringCostWeight + " + " + X2.input + " * " + N3HL1.enteringCostWeight + " + " + X3.input + " * " + N3HL1.enteringCostWeight + " + " + X4.input + " * " + N3HL1.enteringCostWeight + " + " + N3HL1.bias + " = " + N3HL1.x);
        //    Console.WriteLine("1/(1+e^-" + N3HL1.x + ") = " + N3HL1.activationValue);

        //}

        //void N1HL2F()
        //{
        //    previousActivationValue = N1HL1.activationValue;
        //    N1HL2.ActivationvalueCalculation(previousActivationValue);
        //    Console.WriteLine(N1HL1.activationValue + " * " + N1HL2.enteringCostWeight + " = " + N1HL2.weightedSum);

        //    previousActivationValue = N2HL1.activationValue;
        //    N1HL2.ActivationvalueCalculation(previousActivationValue);
        //    Console.WriteLine(N2HL1.activationValue + " * " + N1HL2.enteringCostWeight + " = " + N1HL2.weightedSum);

        //    previousActivationValue = N3HL1.activationValue;
        //    N1HL2.ActivationvalueCalculation(previousActivationValue);
        //    Console.WriteLine(N3HL1.activationValue + " * " + N1HL2.enteringCostWeight + " = " + N1HL2.weightedSum);

        //    N1HL2.SigmoidActivationFunction();
        //    Console.WriteLine(N3HL1.activationValue + " * " + N1HL2.enteringCostWeight + " + " + N3HL1.activationValue + " * " + N1HL2.enteringCostWeight + " + " + N3HL1.activationValue + " * " + N1HL2.enteringCostWeight + " + " + N1HL2.bias + " = " + N1HL2.x);
        //    Console.WriteLine("1/(1+e^-" + N1HL2.x + ") = " + N1HL2.activationValue);
        //}

        //void N2HL2F()
        //{
        //    previousActivationValue = N1HL1.activationValue;
        //    N2HL2.ActivationvalueCalculation(previousActivationValue);
        //    Console.WriteLine(N1HL1.activationValue + " * " + N2HL2.enteringCostWeight + " = " + N2HL2.weightedSum);

        //    previousActivationValue = N2HL1.activationValue;
        //    N2HL2.ActivationvalueCalculation(previousActivationValue);
        //    Console.WriteLine(N2HL1.activationValue + " * " + N2HL2.enteringCostWeight + " = " + N2HL2.weightedSum);

        //    previousActivationValue = N3HL1.activationValue;
        //    N2HL2.ActivationvalueCalculation(previousActivationValue);
        //    Console.WriteLine(N3HL1.activationValue + " * " + N2HL2.enteringCostWeight + " = " + N2HL2.weightedSum);

        //    N2HL2.SigmoidActivationFunction();
        //    Console.WriteLine(N1HL1.activationValue + " * " + N2HL2.enteringCostWeight + " + " + N2HL1.activationValue + " * " + N2HL2.enteringCostWeight + " + " + N3HL1.activationValue + " * " + N2HL2.enteringCostWeight + " + " + N2HL2.bias + " = " + N2HL2.x);
        //    Console.WriteLine("1/(1+e^-" + N2HL2.x + ") = " + N2HL2.activationValue);

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
        //    Console.WriteLine(N1HL1.activationValue + " * " + N3HL2.enteringCostWeight + " + " + N2HL1.activationValue + " * " + N3HL2.enteringCostWeight + " + " + N3HL1.activationValue + " * " + N3HL2.enteringCostWeight + " + " + N3HL2.bias + " = " + N3HL2.x);
        //    Console.WriteLine("1/(1+e^-" + N3HL2.x + ") = " + N3HL2.activationValue);

        //}
        //void O1F()
        //{
        //    previousActivationValue = N1HL2.activationValue;
        //    O1.ActivationvalueCalculation(previousActivationValue);
        //    Console.WriteLine(N1HL2.activationValue + " * " + O1.enteringCostWeight + " = " + O1.weightedSum);

        //    previousActivationValue = N2HL1.activationValue;
        //    O1.ActivationvalueCalculation(previousActivationValue);
        //    Console.WriteLine(N2HL2.activationValue + " * " + O1.enteringCostWeight + " = " + O1.weightedSum);

        //    previousActivationValue = N3HL1.activationValue;
        //    O1.ActivationvalueCalculation(previousActivationValue);
        //    Console.WriteLine(N3HL2.activationValue + " * " + O1.enteringCostWeight + " = " + O1.weightedSum);

        //    O1.SigmoidActivationFunction();
        //    Console.WriteLine(N1HL2.activationValue + " * " + O1.enteringCostWeight + " + " + N2HL2.activationValue + " * " + O1.enteringCostWeight + " + " + N3HL2.activationValue + " * " + O1.enteringCostWeight + " + " + O1.bias + " = " + O1.x);
        //    Console.WriteLine("1/(1+e^-" + O1.x + ") = " + O1.activationValue);

        //}
        //void O2F()
        //{
        //    previousActivationValue = N1HL2.activationValue;
        //    O2.ActivationvalueCalculation(previousActivationValue);
        //    Console.WriteLine(N1HL2.activationValue + " * " + O2.enteringCostWeight + " = " + O2.weightedSum);

        //    previousActivationValue = N2HL1.activationValue;
        //    O2.ActivationvalueCalculation(previousActivationValue);
        //    Console.WriteLine(N2HL2.activationValue + " * " + O2.enteringCostWeight + " = " + O2.weightedSum);

        //    previousActivationValue = N3HL1.activationValue;
        //    O2.ActivationvalueCalculation(previousActivationValue);
        //    Console.WriteLine(N3HL2.activationValue + " * " + O2.enteringCostWeight + " = " + O2.weightedSum);

        //    O2.SigmoidActivationFunction();
        //    Console.WriteLine(N1HL2.activationValue + " * " + O2.enteringCostWeight + " + " + N2HL2.activationValue + " * " + O2.enteringCostWeight + " + " + N3HL2.activationValue + " * " + O2.enteringCostWeight + " + " + O2.bias + " = " + O2.x);
        //    Console.WriteLine("1/(1+e^-" + O2.x + ") = " + O2.activationValue);

        //}

        //public void Initialize()
        //{
        //    Random Y1 = new Random();
        //    Random Y2 = new Random();
        //    Random Y3 = new Random();
        //    Random Y4 = new Random();
        //    Random Y5 = new Random();
        //    Random Y6 = new Random();
        //    Random Y7 = new Random();
        //    Random Y8 = new Random();
        //    Random Y9 = new Random();
        //    Random Y10 = new Random();
        //    Random Y11 = new Random();
        //    Random Y12 = new Random();
        //    Random Y13 = new Random();
        //    Random Y14 = new Random();
        //    Random Y15 = new Random();
        //    Random Y16 = new Random();
        //    Random Y17 = new Random();
        //    Random Y18 = new Random();
        //    Random Y19 = new Random();
        //    Random Y20 = new Random();


        //    //X1.input = 0.1;
        //    //X2.input = 0.1;
        //    //X3.input = 0.1;
        //    //X4.input = 0.1;
        //    X1.input = Y1.NextDouble();
        //    X2.input = Y2.NextDouble();
        //    X3.input = Y3.NextDouble();
        //    X4.input = Y4.NextDouble();

        //    //N1HL1.enteringCostWeight = 0.01;
        //    //N1HL1.bias = 0.5;
        //    N1HL1.enteringCostWeight = Y5.NextDouble();
        //    N1HL1.bias = Y6.NextDouble();

        //    //N2HL1.enteringCostWeight = 0.01;
        //    //N2HL1.bias = 0.5;
        //    N2HL1.enteringCostWeight = Y7.NextDouble();
        //    N2HL1.bias = Y8.NextDouble();

        //    //N3HL1.enteringCostWeight = 0.01;
        //    //N3HL1.bias = 0.5;
        //    N3HL1.enteringCostWeight = Y9.NextDouble();
        //    N3HL1.bias = Y10.NextDouble();

        //    //N1HL2.enteringCostWeight = 0.02;
        //    //N1HL2.bias = 0.4;
        //    N1HL2.enteringCostWeight = Y11.NextDouble();
        //    N1HL2.bias = Y12.NextDouble();

        //    //N2HL2.enteringCostWeight = 0.02;
        //    //N2HL2.bias = 0.4;
        //    N2HL2.enteringCostWeight = Y13.NextDouble();
        //    N2HL2.bias = Y14.NextDouble();

        //    //N3HL2.enteringCostWeight = 0.02;
        //    //N3HL2.bias = 0.4;
        //    N3HL2.enteringCostWeight = Y15.NextDouble();
        //    N3HL2.bias = Y16.NextDouble();

        //    //O1.enteringCostWeight = 0.03;
        //    //O1.bias = 0.3;
        //    O1.enteringCostWeight = Y17.NextDouble();
        //    O1.bias = Y18.NextDouble();
        //    //O1.desiredOutput = 0.1;

        //    //O2.enteringCostWeight = 0.03;
        //    //O2.bias = 0.3;
        //    O2.enteringCostWeight = Y19.NextDouble();
        //    O2.bias = Y20.NextDouble();
        //    //O2.desiredOutput = 0.1;
        //}
        //public void HiddenLayer1()
        //{
        //    Console.WriteLine("Input to Hidden Layer 1");
        //    Console.WriteLine("a(1,1-3) to a(1,1):");
        //    N1HL1F();
        //    Console.WriteLine("\na(1,1-3) to a(1,2):");
        //    N2HL1F();
        //    Console.WriteLine("\na(1,1-3) to a(1,3):");
        //    N3HL1F();
        //    Console.ReadKey();
        //}
        //public void HiddenLayer2()
        //{
        //    Console.WriteLine("\nHidden Layer 1 to Hidden Layer 2");
        //    Console.WriteLine("a(1,1-2) to a(2,1):");
        //    N1HL2F();
        //    Console.WriteLine("\na(1,1-2) to a(2,2):");
        //    N2HL2F();
        //    Console.WriteLine("\na(1,1-2) to a(2,3):");
        //    N3HL2F();
        //    Console.ReadKey();
        //}
        //public void Output()
        //{
        //    Console.WriteLine("\nHidden Layer 2 to Output");
        //    Console.WriteLine("a(1,1-2) to a(3,1):");
        //    O1F();
        //    Console.WriteLine("\na(1,1-2) to a(3,2):");
        //    O2F();
        //    Console.ReadKey();
        //}

    }
}
