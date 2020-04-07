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
        double[,] output = new double[columnMatrixDef, numberOfOutputNeurons];

        public double[,] deltaInputToHiddenLayer1WeightMatrix = new double[HL1NumberofNeurons, numberOfInputNeurons];//3 rows 4 coloums
        public double[,] deltaHiddenLayer1ToHiddenLayer2WeightMatrix = new double[HL2NumberofNeurons, HL1NumberofNeurons];//3 rows 3 coloums
        public double[,] deltaHiddenLayer2ToOutputWeightMatrix = new double[HL2NumberofNeurons, numberOfOutputNeurons];//3 rows 2 coloums
        public double[,] deltaBiasMatrix = new double[columnMatrixDef, numberOfOutputNeurons];//2 rows 1 coloums

        double[,] errorOutput = new double[numberOfOutputNeurons, HL2NumberofNeurons];
        double[,] errorHidden1 = new double[numberOfOutputNeurons, numberOfInputNeurons];

        double[,] ones = new double[numberOfOutputNeurons, 1];

        double[,] inputActivationTranspose = new double[columnMatrixDef, numberOfInputNeurons];//[row,coloumn]
        double[,] outputActivationTranspose = new double[columnMatrixDef, numberOfOutputNeurons];
        double[,] hiddenLayer1ActivationWithoutSig = new double[HL1NumberofNeurons, columnMatrixDef];
        double[,] hiddenLayer1ActivationTranspose = new double[columnMatrixDef, HL1NumberofNeurons];
        double[,] hiddenLayer2ActivationWithoutSig = new double[HL2NumberofNeurons, columnMatrixDef];
        double[,] hiddenLayer2ActivationTranspose = new double[columnMatrixDef, HL2NumberofNeurons];

        public double totalDeltaBias=0;
        public double totalDeltaWeight=0;

        public double totalCostOverAllOutputNeurons=0;
        public double totalCostOverAllTrainingSamples=0;

        public int numberOfTrainingSamples=0;
        public int sampleIterator = 0;

        ActivationFunctions activationFunctions = new ActivationFunctions();
        MatrixOperations matrixOperations = new MatrixOperations();
        PrintOperations printOperations = new PrintOperations();

        //variables needed for 1 Sample, 4 Inputs, 2 Hidden Layers(3 Hidden Neurons each), 2 Outputs
        //public void MultipleInputMultipleHiddenLayerMultipleNeuronMultipleOutput()
        //{
        //    Initialize();
        //    int fCount = Directory.GetFiles("Inputs", "*", SearchOption.TopDirectoryOnly).Length;
        //    Console.Write("Enter Number of Training Samples = ");
        //    numberOfTrainingSamples = fCount;
        //    Console.WriteLine(numberOfTrainingSamples);
        //    for (int j = 0; j < numberOfTrainingSamples; j++)
        //    {
        //        Console.WriteLine("Sample Number = " + j);
        //        printOperations.InputIntializeFormFile(j,numberOfInputNeurons,inputActivation);
        //        printOperations.DesiredOutputIntializeFormFile(j,numberOfOutputNeurons,desiredOutput);
        //        HiddenLayers();
        //        Output(j);
        //        totalCostOverAllTrainingSamples += CostCalculation(j);
        //        BiasUpdate();
        //        WeightUpdate();
        //        Console.WriteLine("Total Cost Over All(" + j + ") Training Samples = " + totalCostOverAllTrainingSamples);
        //        File.WriteAllText("TotalCosts/TotalCost" + j + ".csv",Convert.ToString(totalCostOverAllTrainingSamples));
        //        Console.WriteLine("Average Cost Over All(" + j + ") Training Samples = " + totalCostOverAllTrainingSamples / numberOfTrainingSamples);
        //        File.WriteAllText("AverageCosts/AverageCost" + j + ".csv", Convert.ToString(totalCostOverAllTrainingSamples / numberOfTrainingSamples));
        //        printOperations.BiasUpdateFileStore(j,HL1NumberofNeurons,HL2NumberofNeurons,numberOfOutputNeurons,hiddenLayer1Bias,hiddenLayer2Bias,outputBias);
        //        String path = "Weights/InputToHiddenLayer1/Weight" + j + ".csv";
        //        printOperations.WeightUpdateFileStore(path,HL1NumberofNeurons,numberOfInputNeurons,j,inputToHiddenLayer1WeightMatrix);
        //        path = "Weights/HiddenLayer1ToHiddenLayer2/Weight" + j + ".csv";
        //        printOperations.WeightUpdateFileStore(path, HL2NumberofNeurons, HL1NumberofNeurons,j,hiddenLayer1ToHiddenLayer2WeightMatrix);
        //        path = "Weights/HiddenLayer2ToOutput/Weight" + j + ".csv";
        //        printOperations.WeightUpdateFileStore(path, HL2NumberofNeurons, numberOfOutputNeurons,j,hiddenLayer2ToOutputWeightMatrix);
        //        sampleIterator++;
        //    }
        //    Console.ReadKey();

        //}
        public void Train()
        {
            Initialize();
            Calculation();
            Print();
            Console.ReadKey();
        }
        void Initialize()
        {
            WeightMatrixInitialize();
            BiasInitialize();
        }
        void Calculation()
        {
            hiddenLayer1ActivationWithoutSig = matrixOperations.MatrixMultiplication(inputToHiddenLayer1WeightMatrix, HL1NumberofNeurons, numberOfInputNeurons, inputActivation, columnMatrixDef);
            for (int i = 0; i < HL1NumberofNeurons; i++)
            {
                hiddenLayer1Activation[i, 0] = activationFunctions.Sigmoid(hiddenLayer1ActivationWithoutSig[i, 0]);
                hiddenLayer1ActivationTranspose[0, i] = hiddenLayer1Activation[i, 0];
            }
            hiddenLayer2ActivationWithoutSig = matrixOperations.MatrixMultiplication(hiddenLayer1ToHiddenLayer2WeightMatrix, HL2NumberofNeurons, HL1NumberofNeurons, hiddenLayer1Activation, columnMatrixDef);
            for (int i = 0; i < HL2NumberofNeurons; i++)
            {
                hiddenLayer2Activation[i, 0] = activationFunctions.Sigmoid(hiddenLayer2ActivationWithoutSig[i, 0]);
                hiddenLayer2ActivationTranspose[0, i] = hiddenLayer2Activation[i, 0];
            }
            output = matrixOperations.MatrixMultiplication(hiddenLayer2ToOutputWeightMatrix, numberOfOutputNeurons, HL2NumberofNeurons, hiddenLayer2Activation, columnMatrixDef);
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                outputActivation[i, 0] = activationFunctions.Sigmoid(output[i, 0]);
                outputActivationTranspose[0, i] = outputActivation[i, 0];
            }
            errorOutput = matrixOperations.MatrixMultiplication(errorOutput, numberOfOutputNeurons, columnMatrixDef, outputActivationTranspose, numberOfOutputNeurons);
            errorOutput = matrixOperations.MatrixMultiplication(errorOutput, numberOfOutputNeurons, columnMatrixDef, matrixOperations.MatrixSubstraction(ones, numberOfOutputNeurons, columnMatrixDef, output), numberOfOutputNeurons);
            deltaHiddenLayer2ToOutputWeightMatrix = matrixOperations.MatrixMultiplication(errorOutput, numberOfOutputNeurons, columnMatrixDef, hiddenLayer1ActivationTranspose, numberOfOutputNeurons);
        }
        void WeightMatrixInitialize()
        {
            matrixOperations.InitializeMatrix(inputToHiddenLayer1WeightMatrix,HL1NumberofNeurons,numberOfInputNeurons,weightInitialValue);
            matrixOperations.InitializeMatrix(hiddenLayer1ToHiddenLayer2WeightMatrix,HL2NumberofNeurons,HL1NumberofNeurons,weightInitialValue);
            matrixOperations.InitializeMatrix(hiddenLayer2ToOutputWeightMatrix,HL2NumberofNeurons,numberOfOutputNeurons,weightInitialValue);
        }
        void BiasInitialize()
        {
            matrixOperations.InitializeMatrix(hiddenLayer1Bias,HL1NumberofNeurons,columnMatrixDef,biasInitialValue);
            matrixOperations.InitializeMatrix(hiddenLayer2Bias, HL2NumberofNeurons, columnMatrixDef, biasInitialValue);
            matrixOperations.InitializeMatrix(outputActivation, numberOfOutputNeurons, columnMatrixDef, biasInitialValue);
        }
        void Print()
        {
            printOperations.PrintMatrix("Weight Matrix Input To Hidden Layer 1:", HL1NumberofNeurons, numberOfInputNeurons, inputToHiddenLayer1WeightMatrix);
            printOperations.PrintMatrixFile("Weight Input To Hidden Layer 1.csv", HL1NumberofNeurons, numberOfInputNeurons, inputToHiddenLayer1WeightMatrix);
            Console.WriteLine();

            printOperations.PrintMatrix("Matrix Input Activation:", numberOfInputNeurons, columnMatrixDef, inputActivation);
            printOperations.PrintMatrixFile("InputActivation.csv", numberOfInputNeurons, columnMatrixDef, inputActivation);
            Console.WriteLine();

            printOperations.PrintMatrix("Matrix Hidden Layer 1 Activation Without Sig:", HL1NumberofNeurons, columnMatrixDef, hiddenLayer1ActivationWithoutSig);
            printOperations.PrintMatrixFile("Hidden Layer 1 Activation Without Sigmoid.csv", HL1NumberofNeurons, columnMatrixDef, hiddenLayer1ActivationWithoutSig);
            Console.WriteLine();

            printOperations.PrintMatrix("Matrix Hidden Layer 1 Activation With Sig:", HL1NumberofNeurons, columnMatrixDef, hiddenLayer1Activation);
            printOperations.PrintMatrixFile("Hidden Layer 1 Activation With Sigmoid.csv", HL1NumberofNeurons, columnMatrixDef, hiddenLayer1Activation);
            Console.WriteLine();

            printOperations.PrintMatrix("Weight Matrix Hidden Layer 1 To Hidden Layer 2:", HL2NumberofNeurons, HL1NumberofNeurons, hiddenLayer1ToHiddenLayer2WeightMatrix);
            printOperations.PrintMatrixFile("Weight Hidden Layer 1 To Hidden Layer 2.csv", HL2NumberofNeurons, HL1NumberofNeurons, hiddenLayer1ToHiddenLayer2WeightMatrix);
            Console.WriteLine();

            printOperations.PrintMatrix("Matrix Hidden Layer 2 Activation Without Sig:", HL2NumberofNeurons, columnMatrixDef, hiddenLayer1ActivationWithoutSig);
            printOperations.PrintMatrixFile("Hidden Layer 2 Activation Without Sigmoid.csv", HL2NumberofNeurons, columnMatrixDef, hiddenLayer1ActivationWithoutSig);
            Console.WriteLine();

            printOperations.PrintMatrix("Matrix Hidden Layer 2 Activation With Sig:", HL2NumberofNeurons, columnMatrixDef, hiddenLayer1Activation);
            printOperations.PrintMatrixFile("Hidden Layer 2 Activation With Sigmoid.csv", HL2NumberofNeurons, columnMatrixDef, hiddenLayer1Activation);
            Console.WriteLine();

            printOperations.PrintMatrix("Weight Matrix Hidden Layer 2 To Output:", numberOfOutputNeurons, HL2NumberofNeurons, hiddenLayer2ToOutputWeightMatrix);
            printOperations.PrintMatrixFile("Weight Hidden Layer 2 To Output.csv", numberOfOutputNeurons, HL2NumberofNeurons, hiddenLayer2ToOutputWeightMatrix);
            Console.WriteLine();

            printOperations.PrintMatrix("Matrix Output Layer Activation Without Sig:", numberOfOutputNeurons, columnMatrixDef, output);
            printOperations.PrintMatrixFile("Output Layer Activation Without Sigmoid.csv", numberOfOutputNeurons, columnMatrixDef, output);
            Console.WriteLine();

            printOperations.PrintMatrix("Output Layer Activation With Sig:", numberOfOutputNeurons, columnMatrixDef, outputActivation);
            printOperations.PrintMatrixFile("Output Layer Activation With Sigmoid.csv", numberOfOutputNeurons, columnMatrixDef, outputActivation);
            Console.WriteLine();

        }
        double GetRandomNumber(double min, double max, int randomizer)
        {
            double output;
            Random X = new Random();
            output = X.NextDouble() * ((max - min) / min) + (randomizer * 0.0001);
            return (output);
        }

    }
}
