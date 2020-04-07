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

        //static int inputNeurons = IN;
        //static int hiddenLayer1Neurons = HL1N;
        //static int hiddenLayer2Neurons = HL2N;
        //static int outputNeurons = ON;
        //static int columnMatrix = CM;

        static int IN = 784;
        static int HL1N = 16;
        static int HL2N = 16;
        static int ON = 10;
        static int CM = 1;

        public double weightInitialValue = 0.001;
        public double biasInitialValue = 0;
        public double learningRate = 0.01;


        double[,] outputActivationTranspose = new double[CM, ON];
        double[,] EO = new double[ON, CM];
        double[,] EOTranspose = new double[CM, ON];
        double[,] EH2 = new double[ON, HL2N];
        double[,] EH1 = new double[CM, CM];
        double[,] EHI = new double[HL1N, IN];


        public double[,] weightIH1 = new double[HL1N, IN];//3 rows 4 coloums
        public double[,] weightH1H2 = new double[HL2N, HL1N];//3 rows 3 coloums
        public double[,] weightH2O = new double[ON, HL2N];//3 rows 2 coloums

        public double[,] inputActivation = new double[IN, CM];//4 rows 1 coloums
        public double[,] hiddenLayer1Activation = new double[HL1N, CM];//3 rows 1 coloums
        public double[,] hiddenLayer2Activation = new double[HL2N, CM];//3 rows 1 coloums
        public double[,] outputActivation = new double[ON, CM];//2 rows 1 coloums
        //Checked

        public double[,] hiddenLayer1Bias = new double[HL1N, CM];//3 rows 1 coloums
        public double[,] hiddenLayer2Bias = new double[HL2N, CM];//3 rows 1 coloums
        public double[,] outputBias = new double[ON, CM];//2 rows 1 coloums

        
        public double[,] desiredOutput = new double[ON, CM];//2 rows 1 coloums
        double[,] output = new double[ON,CM];

        public double[,] deltaweightIH1 = new double[HL1N, IN];//3 rows 4 coloums
        public double[,] deltaweightH1H2 = new double[HL2N, HL1N];//3 rows 3 coloums
        public double[,] deltaweightH2O = new double[ON,HL2N];//3 rows 2 coloums
        public double[,] deltaBiasMatrix = new double[CM, ON];//2 rows 1 coloums

        double[,] errorOutput = new double[ON, CM];
        double[,] errorHidden1 = new double[ON, IN];

        double[,] inputActivationTranspose = new double[CM, IN];//[row,coloumn]
        double[,] hiddenLayer1ActivationWithoutSig = new double[HL1N, CM];
        double[,] hiddenLayer1ActivationTranspose = new double[CM, HL1N];
        double[,] hiddenLayer2ActivationWithoutSig = new double[HL2N, CM];
        double[,] hiddenLayer2ActivationTranspose = new double[CM, HL2N];

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
            //BiasInitialize();
        }
        void Calculation()
        {
            hiddenLayer1ActivationWithoutSig = matrixOperations.MatrixMultiplication(weightIH1, HL1N, IN, inputActivation, CM);
            for (int i = 0; i < HL1N; i++)
            {
                hiddenLayer1Activation[i, 0] = activationFunctions.Sigmoid(hiddenLayer1ActivationWithoutSig[i, 0]);
                hiddenLayer1ActivationTranspose[0, i] = hiddenLayer1Activation[i, 0];
            }
            hiddenLayer2ActivationWithoutSig = matrixOperations.MatrixMultiplication(weightH1H2, HL2N, HL1N, hiddenLayer1Activation, CM);
            for (int i = 0; i < HL2N; i++)
            {
                hiddenLayer2Activation[i, 0] = activationFunctions.Sigmoid(hiddenLayer2ActivationWithoutSig[i, 0]);
                hiddenLayer2ActivationTranspose[0, i] = hiddenLayer2Activation[i, 0];
            }
            output = matrixOperations.MatrixMultiplication(weightH2O, ON, HL2N, hiddenLayer2Activation, CM);
            for (int i = 0; i < ON; i++)
            {
                outputActivation[i, 0] = activationFunctions.Sigmoid(output[i, 0]);
                outputActivationTranspose[0, i] = outputActivation[i, 0];
            }
            errorOutput = matrixOperations.MatrixMultiplication(errorOutput, ON, CM, outputActivationTranspose, ON);
            //errorOutput = matrixOperations.MatrixMultiplication(errorOutput, ON, CM, matrixOperations.MatrixSubstraction(matrixOperations.Ones(ON, CM), ON, CM, output), ON);
            //deltaweightH2O = matrixOperations.MatrixMultiplication(errorOutput, ON, CM, hiddenLayer1ActivationTranspose, ON);
        }
        void WeightMatrixInitialize()
        {
            matrixOperations.InitializeMatrix(weightIH1,HL1N,IN,weightInitialValue);
            matrixOperations.InitializeMatrix(weightH1H2,HL2N,HL1N,weightInitialValue);
            matrixOperations.InitializeMatrix(weightH2O,ON,HL2N,weightInitialValue);
        }
        void BiasInitialize()
        {
            matrixOperations.InitializeMatrix(hiddenLayer1Bias,HL1N,CM,biasInitialValue);
            matrixOperations.InitializeMatrix(hiddenLayer2Bias, HL2N, CM, biasInitialValue);
            matrixOperations.InitializeMatrix(outputActivation, ON, CM, biasInitialValue);
        }
        void Print()
        {
            //printOperations.PrintMatrix("Weight Matrix Input To Hidden Layer 1:", HL1N, IN, weightIH1);
            //printOperations.PrintMatrixFile("Weight Input To Hidden Layer 1.csv", HL1N, IN, weightIH1);
            Console.WriteLine();

            printOperations.PrintMatrixToConsole("Matrix Input Activation:", IN, CM, inputActivation);
            printOperations.PrintMatrixFile("InputActivation.csv", IN, CM, inputActivation);
            Console.WriteLine();

            printOperations.PrintMatrixToConsole("Matrix Hidden Layer 1 Activation Without Sig:", HL1N, CM, hiddenLayer1ActivationWithoutSig);
            printOperations.PrintMatrixFile("Hidden Layer 1 Activation Without Sigmoid.csv", HL1N, CM, hiddenLayer1ActivationWithoutSig);
            Console.WriteLine();

            printOperations.PrintMatrixToConsole("Matrix Hidden Layer 1 Activation With Sig:", HL1N, CM, hiddenLayer1Activation);
            printOperations.PrintMatrixFile("Hidden Layer 1 Activation With Sigmoid.csv", HL1N, CM, hiddenLayer1Activation);
            Console.WriteLine();

            //printOperations.PrintMatrix("Weight Matrix Hidden Layer 1 To Hidden Layer 2:", HL2N, HL1N, weightH1H2);
            printOperations.PrintMatrixFile("Weight Hidden Layer 1 To Hidden Layer 2.csv", HL2N, HL1N, weightH1H2);
            Console.WriteLine();

            printOperations.PrintMatrixToConsole("Matrix Hidden Layer 2 Activation Without Sig:", HL2N, CM, hiddenLayer1ActivationWithoutSig);
            printOperations.PrintMatrixFile("Hidden Layer 2 Activation Without Sigmoid.csv", HL2N, CM, hiddenLayer1ActivationWithoutSig);
            Console.WriteLine();

            printOperations.PrintMatrixToConsole("Matrix Hidden Layer 2 Activation With Sig:", HL2N, CM, hiddenLayer1Activation);
            printOperations.PrintMatrixFile("Hidden Layer 2 Activation With Sigmoid.csv", HL2N, CM, hiddenLayer1Activation);
            Console.WriteLine();

            //printOperations.PrintMatrix("Weight Matrix Hidden Layer 2 To Output:", ON, HL2NumberofNeurons, weightH2O);
            printOperations.PrintMatrixFile("Weight Hidden Layer 2 To Output.csv", ON, HL2N, weightH2O);
            Console.WriteLine();

            printOperations.PrintMatrixToConsole("Matrix Output Layer Activation Without Sig:", ON, CM, output);
            printOperations.PrintMatrixFile("Output Layer Activation Without Sigmoid.csv", ON, CM, output);
            Console.WriteLine();

            printOperations.PrintMatrixToConsole("Output Layer Activation With Sig:", ON, CM, outputActivation);
            printOperations.PrintMatrixFile("Output Layer Activation With Sigmoid.csv", ON, CM, outputActivation);
            Console.WriteLine();

            printOperations.PrintMatrixToConsole("Error Output:", ON, CM, errorOutput);
            printOperations.PrintMatrixFile("Error Output.csv", ON, CM, errorOutput);
            Console.WriteLine();

        }
        double GetRandomNumber(double min, double max)
        {
            double output;
            Random X = new Random();
            output = X.NextDouble() * ((max - min) / min);
            return (output);
        }

    }
}
