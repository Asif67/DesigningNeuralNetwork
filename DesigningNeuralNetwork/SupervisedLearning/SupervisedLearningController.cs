using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DesigningNeuralNetwork.NonLinearity;
using DesigningNeuralNetwork.Operations;
using System.Windows.Media;
using System.IO;


namespace DesigningNeuralNetwork.SupervisedLearning
{
    class SupervisedLearningController
    {
        //variables needed for 1 Sample, 4 Inputs, 2 Hidden Layers(3 Hidden Neurons each), 2 Outputs

        static int numberOfInputNeurons = 784;
        static int columnMatrixDef = 1;
        static int HL1NumberofNeurons = 16;
        static int HL2NumberofNeurons = 16;
        static int numberOfOutputNeurons = 10;
        public double weightInitialValue = 0.001;
        public double biasInitialValue = 0;
        public double learningRate = 0.01;

        public double[,] inputToHiddenLayer1WeightMatrix = new double[HL1NumberofNeurons, numberOfInputNeurons];//3 rows 4 coloums
        public double[,] hiddenLayer1ToHiddenLayer2WeightMatrix = new double[HL2NumberofNeurons, HL1NumberofNeurons];//3 rows 3 coloums
        public double[,] hiddenLayer2ToOutputWeightMatrix = new double[HL2NumberofNeurons, numberOfOutputNeurons];//3 rows 2 coloums

        public double[,] inputActivation = new double[numberOfInputNeurons, columnMatrixDef];//4 rows 1 coloums
        public double[,] hiddenLayer1Activation = new double[HL1NumberofNeurons, columnMatrixDef];//3 rows 1 coloums
        public double[,] hiddenLayer2Activation = new double[HL2NumberofNeurons, columnMatrixDef];//3 rows 1 coloums
        public double[,] outputActivation = new double[numberOfOutputNeurons, columnMatrixDef];//2 rows 1 coloums

        public double[,] hiddenLayer1Bias = new double[HL1NumberofNeurons, columnMatrixDef];//3 rows 1 coloums
        public double[,] hiddenLayer2Bias = new double[HL2NumberofNeurons, columnMatrixDef];//3 rows 1 coloums
        public double[,] outputBias = new double[numberOfOutputNeurons, columnMatrixDef];//2 rows 1 coloums

        public double[,] z1 = new double[HL1NumberofNeurons, columnMatrixDef];//3 rows 1 coloums
        public double[,] z2 = new double[HL2NumberofNeurons, columnMatrixDef];//3 rows 1 coloums
        public double[,] z3 = new double[numberOfOutputNeurons, columnMatrixDef];//2 rows 1 coloums
        public double[,] z3Prime = new double[numberOfOutputNeurons, columnMatrixDef];//2 rows 1 coloums

        public double[,] desiredOutput = new double[numberOfOutputNeurons, columnMatrixDef];//2 rows 1 coloums
        public double[,] cost = new double[numberOfOutputNeurons, columnMatrixDef];//2 rows 1 coloums

        public double[,] deltaWeightMatrix = new double[columnMatrixDef, numberOfOutputNeurons];//2 rows 1 coloums
        public double[,] deltaBiasMatrix = new double[columnMatrixDef, numberOfOutputNeurons];//2 rows 1 coloums

        public double totalDeltaBias=0;
        public double totalDeltaWeight=0;

        public double totalCostOverAllOutputNeurons=0;
        public double totalCostOverAllTrainingSamples=0;

        public int numberOfTrainingSamples=0;
        public int sampleIterator = 0;

        ActivationFunctions activationFunctions = new ActivationFunctions();
        FirstOrderDerivatives firstOrderDerivatives = new FirstOrderDerivatives();
        MatrixOperations matrixOperations = new MatrixOperations();
        FileOperations fileOperations = new FileOperations();

        //variables needed for 1 Sample, 4 Inputs, 2 Hidden Layers(3 Hidden Neurons each), 2 Outputs
        public void MultipleInputMultipleHiddenLayerMultipleNeuronMultipleOutput()
        {
            Initialize();
            int fCount = Directory.GetFiles("Inputs", "*", SearchOption.TopDirectoryOnly).Length;
            Console.Write("Enter Number of Training Samples = ");
            numberOfTrainingSamples = fCount;
            Console.WriteLine(numberOfTrainingSamples);
            for (int j = 0; j < numberOfTrainingSamples; j++)
            {
                Console.WriteLine("Sample Number = " + j);
                fileOperations.InputIntializeFormFile(j,numberOfInputNeurons,inputActivation);
                fileOperations.DesiredOutputIntializeFormFile(j,numberOfOutputNeurons,desiredOutput);
                HiddenLayers();
                Output(j);
                ActivationPrint();
                DesiredOutputPrint();
                totalCostOverAllTrainingSamples += CostCalculation(j);
                BiasUpdate();
                WeightUpdate();
                Console.WriteLine("Total Cost Over All(" + j + ") Training Samples = " + totalCostOverAllTrainingSamples);
                File.WriteAllText("TotalCosts/TotalCost" + j + ".csv",Convert.ToString(totalCostOverAllTrainingSamples));
                Console.WriteLine("Average Cost Over All(" + j + ") Training Samples = " + totalCostOverAllTrainingSamples / numberOfTrainingSamples);
                File.WriteAllText("AverageCosts/AverageCost" + j + ".csv", Convert.ToString(totalCostOverAllTrainingSamples / numberOfTrainingSamples));
                fileOperations.BiasUpdateFileStore(j,HL1NumberofNeurons,HL2NumberofNeurons,numberOfOutputNeurons,hiddenLayer1Bias,hiddenLayer2Bias,outputBias);
                String path = "Weights/InputToHiddenLayer1/Weight" + j + ".csv";
                fileOperations.WeightUpdateFileStore(path,HL1NumberofNeurons,numberOfInputNeurons,j,inputToHiddenLayer1WeightMatrix);
                path = "Weights/HiddenLayer1ToHiddenLayer2/Weight" + j + ".csv";
                fileOperations.WeightUpdateFileStore(path, HL2NumberofNeurons, HL1NumberofNeurons,j,hiddenLayer1ToHiddenLayer2WeightMatrix);
                path = "Weights/HiddenLayer2ToOutput/Weight" + j + ".csv";
                fileOperations.WeightUpdateFileStore(path, HL2NumberofNeurons, numberOfOutputNeurons,j,hiddenLayer2ToOutputWeightMatrix);
                sampleIterator++;
            }
            Console.ReadKey();

        }
        void Initialize()
        {
            WeightMatrixInitialize();
            BiasInitialize();
        }
        void HiddenLayers()
        {
            HiddenLayer1();
            HiddenLayer2();
        }
        void HiddenLayer1()
        {

            Console.WriteLine("\nHidden Layer 1:");
            z1 = matrixOperations.MatrixMultiplication(inputToHiddenLayer1WeightMatrix, HL1NumberofNeurons, numberOfInputNeurons, inputActivation, numberOfInputNeurons, columnMatrixDef);
            z1 = matrixOperations.MatrixAddition(z1, HL1NumberofNeurons, columnMatrixDef, hiddenLayer1Bias, HL1NumberofNeurons, columnMatrixDef);
            Console.WriteLine("\nAfter SigmoidActivation Function application:");
            for (int i = 0; i < HL1NumberofNeurons; i++)
            {
                z1[i, 0] = activationFunctions.Sigmoid(z1[i, 0]);
                Console.WriteLine(z1[i, 0]);
            }
            hiddenLayer1Activation = z1;
            //Console.ReadKey();
        }
        void HiddenLayer2()
        {
            Console.WriteLine("\nHidden Layer 2:");
            z2 = matrixOperations.MatrixMultiplication(hiddenLayer1ToHiddenLayer2WeightMatrix, HL2NumberofNeurons, HL1NumberofNeurons, hiddenLayer1Activation, HL2NumberofNeurons, columnMatrixDef);
            z2 = matrixOperations.MatrixAddition(z2, HL2NumberofNeurons, columnMatrixDef, hiddenLayer2Bias, HL2NumberofNeurons, columnMatrixDef);
            Console.WriteLine("\nAfter SigmoidActivation Function application:");
            for (int i = 0; i < HL2NumberofNeurons; i++)
            {
                z2[i, 0] = activationFunctions.Sigmoid(z2[i, 0]);
                Console.WriteLine(z2[i, 0]);
            }
            hiddenLayer2Activation = z2;
            //Console.ReadKey();
        }
        void Output(int sampleNumber)
        {
            Console.WriteLine("\nOutput:");
            z3 = matrixOperations.MatrixMultiplication(hiddenLayer2ToOutputWeightMatrix, HL2NumberofNeurons, numberOfOutputNeurons, outputActivation, numberOfOutputNeurons, columnMatrixDef);
            z3 = matrixOperations.MatrixAddition(z3, numberOfOutputNeurons, columnMatrixDef, outputBias, numberOfOutputNeurons, columnMatrixDef);
            z3Prime = z3; // storing zL for weight and bias update
            Console.WriteLine("\nAfter SigmoidActivation Function application:");
            File.WriteAllText("NetworkOutputs/NetworkOuput" + sampleNumber + ".csv", "");
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                z3[i, 0] = activationFunctions.Sigmoid(z3[i, 0]);
                File.AppendAllText("NetworkOutputs/NetworkOuput" + sampleNumber + ".csv", Convert.ToString(z3[i, 0]) + ",");
                Console.WriteLine(z3[i, 0]);
            }
            outputActivation = z3;
            //Console.ReadKey();
        }
        void WeightMatrixInitialize()
        {

            Console.WriteLine("Input to Hidden Layer 1 Weight Matrix\n\n");
            //Initialize Weight Matrix
            for (int i = 0; i < HL1NumberofNeurons; i++)
            {
                for (int j = 0; j < numberOfInputNeurons; j++)
                {
                    Random X1 = new Random();
                    inputToHiddenLayer1WeightMatrix[i, j] = weightInitialValue;
                    Console.Write("IH1WM(" + i + "," + j + ") = "+"{0}\t", inputToHiddenLayer1WeightMatrix[i, j]);
                }
                Console.WriteLine("\n");
            }
            Console.WriteLine("Hidden Layer 1 to Hidden Layer 2 Weight Matrix\n\n");
            for (int i = 0; i < HL2NumberofNeurons; i++)
            {
                for (int j = 0; j < HL1NumberofNeurons; j++)
                {
                    Random X2 = new Random();
                    hiddenLayer1ToHiddenLayer2WeightMatrix[i, j] = weightInitialValue;
                    Console.Write("H1H2WM(" + i + "," + j + ") = " + "{0}\t", hiddenLayer1ToHiddenLayer2WeightMatrix[i, j]);
                }
                Console.WriteLine("\n");
            }
            Console.WriteLine("Hidden Layer 2 to Output Weight Matrix\n\n");
            for (int i = 0; i < HL2NumberofNeurons; i++)
            {
                for (int j = 0; j < numberOfOutputNeurons; j++)
                {
                    Random X3 = new Random();
                    hiddenLayer2ToOutputWeightMatrix[i, j] = weightInitialValue;
                    Console.Write("H2OWM(" + i + "," + j + ") = " + "{0}\t", hiddenLayer2ToOutputWeightMatrix[i, j]);
                }
                Console.WriteLine("\n");
            }
            //Initialize Weight Matrix
        }
        void ActivationPrint()
        {
            
            //Show Activation Matrix
            for (int i = 0; i < numberOfInputNeurons; i++)
            {
                Console.WriteLine("IA(" + 0 + "," + i + ") = " + inputActivation[i,0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < HL1NumberofNeurons; i++)
            {
                Console.WriteLine("H1A(" + 0 + "," + i + ") = " + hiddenLayer1Activation[i,0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < HL2NumberofNeurons; i++)
            {
                Console.WriteLine("H2A(" + 0 + "," + i + ") = " + hiddenLayer2Activation[i,0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                Console.WriteLine("OA(" + 0 + "," + i + ") = " + outputActivation[i, 0]);
            }
            Console.WriteLine("\n");
            //Show Activation Matrix
        }
        void BiasInitialize()
        {
            
            //Initialize bias Matrix
            for (int i = 0; i < HL1NumberofNeurons; i++)
            {
                Random X1 = new Random();
                hiddenLayer1Bias[i, 0] = biasInitialValue;
                Console.WriteLine("H1B(" + 0 + "," + i + ") = " + hiddenLayer1Bias[i,0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < HL2NumberofNeurons; i++)
            {
                Random X2 = new Random();
                hiddenLayer2Bias[i, 0] = biasInitialValue;
                Console.WriteLine("H2B(" + 0 + "," + i + ") = " + hiddenLayer2Bias[i, 0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                Random X3 = new Random();
                outputBias[i, 0] = biasInitialValue;
                Console.WriteLine("OB(" + 0 + "," + i + ") = " + outputBias[i, 0]);
            }
            Console.WriteLine("\n");
            //Initialize bias Matrix
        }
        void DesiredOutputPrint()
        {
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                //desiredOutput[i, 0] = 1;
                Console.WriteLine(desiredOutput[i, 0]);
            }
            //Console.ReadKey();
        }
        double CostCalculation(int sampleNumber)
        {
            Console.WriteLine("\nCost(desiredoutput - predicted output)^2:");
            //for (int i = 0; i < numberOfOutputNeurons; i++)
            //{
            //    cost[i, 0] = Math.Pow((cost[i,0]), 2);
            //    totalCostOverAllOutputNeurons += cost[i, 0];
            //    Console.WriteLine(cost[i, 0]);
            //}
            //Console.WriteLine("Total Cost of " + numberOfTrainingSamples + " Training Samples:");
            //Console.WriteLine(totalCostOverAllOutputNeurons);
            //Console.WriteLine("Average Cost of " + numberOfTrainingSamples + " Training Samples:");
            //Console.WriteLine(totalCostOverAllOutputNeurons / numberOfInputNeurons);
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
            cost = matrixOperations.MatrixSubstraction(z3, numberOfOutputNeurons, columnMatrixDef, desiredOutput, numberOfOutputNeurons, columnMatrixDef);
            Console.WriteLine("\nCost(Simplified Chain rule applied):");
            File.WriteAllText("CostOverAllOutputNeuron/CostOverAllOutputNeuronsTrainingSample" + sampleNumber + ".csv", "");
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                //cost[i, 0] = (-2) * numberOfNeuronsInPreviousLayer * cost[i, 0];
                File.AppendAllText("CostOverAllOutputNeuron/CostOverAllOutputNeuronsTrainingSample" + sampleNumber + ".csv",Convert.ToString(cost[i,0])+",");
                totalCostOverAllOutputNeurons += cost[i, 0];
                Console.WriteLine(cost[i, 0]);
            }
            Console.WriteLine("Total Cost of " + numberOfTrainingSamples + " Training Samples:");
            Console.WriteLine(totalCostOverAllOutputNeurons);
            Console.WriteLine("Average Cost of " + numberOfTrainingSamples + " Training Samples:");
            Console.WriteLine(totalCostOverAllOutputNeurons / numberOfInputNeurons);
            return (totalCostOverAllOutputNeurons);
            //Console.ReadKey();
        }
        double deltaWeightCalculation()
        {
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                deltaWeightMatrix[0, i] = (-2) * HL2NumberofNeurons * cost[i, 0] * learningRate;//Gradient Desent
            }
            //deltaWeightMatrix =  MD(cost, numberOfOutputNeurons, columnMatrixDef, TM(z3, numberOfOutputNeurons, columnMatrixDef), columnMatrixDef, numberOfOutputNeurons);
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                totalDeltaWeight += deltaWeightMatrix[0, i];
                //Console.WriteLine(cost[i, 0]);
            }
            return (totalDeltaWeight);
        }
        void WeightUpdate()
        {
            Console.WriteLine("Input to Hidden Layer 1 Weight Matrix\n\n");
            //Initialize Weight Matrix
            for (int i = 0; i < HL1NumberofNeurons; i++)
            {
                for (int j = 0; j < numberOfInputNeurons; j++)
                {
                    Random X1 = new Random();
                    inputToHiddenLayer1WeightMatrix[i, j] -= deltaWeightCalculation();
                    //inputToHiddenLayer1WeightMatrix[i, j] = 0.01;
                    //Console.WriteLine("IH1WM(" + i + "," + j + ") = " + inputToHiddenLayer1WeightMatrix[i, j]);
                    Console.Write("IH1WM(" + i + "," + j + ") = " + "{0}\t", inputToHiddenLayer1WeightMatrix[i, j]);
                }
                Console.WriteLine("\n");
            }
            Console.WriteLine("Hidden Layer 1 to Hidden Layer 2 Weight Matrix\n\n");
            for (int i = 0; i < HL2NumberofNeurons; i++)
            {
                for (int j = 0; j < HL1NumberofNeurons; j++)
                {
                    Random X2 = new Random();
                    hiddenLayer1ToHiddenLayer2WeightMatrix[i, j] -= deltaWeightCalculation();
                    //hiddenLayer1ToHiddenLayer2WeightMatrix[i, j] = 0.02;
                    //Console.WriteLine("H1H2WM(" + i + "," + j + ") = " + hiddenLayer1ToHiddenLayer2WeightMatrix[i, j]);
                    Console.Write("H1H2WM(" + i + "," + j + ") = " + "{0}\t", hiddenLayer1ToHiddenLayer2WeightMatrix[i, j]);
                }
                Console.WriteLine("\n");
            }
            Console.WriteLine("Hidden Layer 2 to Output Weight Matrix\n\n");
            for (int i = 0; i < HL2NumberofNeurons; i++)
            {
                for (int j = 0; j < numberOfOutputNeurons; j++)
                {
                    Random X3 = new Random();
                    hiddenLayer2ToOutputWeightMatrix[i, j] -= deltaWeightCalculation();
                    //hiddenLayer2ToOutputWeightMatrix[i, j] = 0.03;
                    //Console.WriteLine("H2OWM(" + i + "," + j + ") = " + hiddenLayer2ToOutputWeightMatrix[i, j]);
                    Console.Write("H2OWM(" + i + "," + j + ") = " + "{0}\t", hiddenLayer2ToOutputWeightMatrix[i, j]);
                }
                Console.WriteLine("\n");
            }
        }
        double deltaBiasCalculation()
        {
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                for(int j = 0; j < HL2NumberofNeurons; j++)
                {
                    deltaBiasMatrix[0, i] = (-2) * hiddenLayer2Bias[j, 0] * cost[i, 0] * learningRate;//Gradient Desent
                }
            }
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                totalDeltaBias += deltaBiasMatrix[0, i];
                //Console.WriteLine(cost[i, 0]);
            }
            return (totalDeltaBias);
        }
        void BiasUpdate()
        {
            //Initialize bias Matrix
            for (int i = 0; i < HL1NumberofNeurons; i++)
            {
                Random X1 = new Random();
                hiddenLayer1Bias[i, 0] -= deltaBiasCalculation();
                //hiddenLayer1Bias[i, 0] = 0.001;
                Console.WriteLine("H1B(" + 0 + "," + i + ") = " + hiddenLayer1Bias[i, 0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < HL2NumberofNeurons; i++)
            {
                Random X2 = new Random();
                hiddenLayer2Bias[i, 0] -= deltaBiasCalculation();
                //hiddenLayer2Bias[i, 0] = 0.002;
                Console.WriteLine("H2B(" + 0 + "," + i + ") = " + hiddenLayer2Bias[i, 0]);
            }
            Console.WriteLine("\n");
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                Random X3 = new Random();
                outputBias[i, 0] -= deltaBiasCalculation();
                //outputBias[i, 0] = 0.003;
                Console.WriteLine("OB(" + 0 + "," + i + ") = " + outputBias[i, 0]);
            }
            Console.WriteLine("\n");
            //Initialize bias Matrix
        }
        
    }
}
