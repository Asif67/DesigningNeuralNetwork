using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using System.IO;

namespace DesigningNeuralNetwork
{
    class New
    {
        //2->3
        static int numberOfInputNeurons = 3;
        static int numberOfHiddenLayer1Neurons = 3;
        static int numberOfOutputNeurons = 2;
        static double learningRate = 0.01;

        double[,] inputActivation = new double[numberOfInputNeurons, 1];//[row,coloumn]
        double[,] inputActivationTranspose = new double[1,numberOfInputNeurons];//[row,coloumn]

        double[,] hiddenLayer1Activation = new double[numberOfHiddenLayer1Neurons, 1];
        double[,] hiddenLayer1ActivationWithoutSig = new double[numberOfHiddenLayer1Neurons, 1];
        double[,] hiddenLayer1ActivationTranspose = new double[1,numberOfHiddenLayer1Neurons];

        double[,] outputActivation = new double[numberOfOutputNeurons, 1];
        double[,] outputActivationTranspose = new double[1, numberOfOutputNeurons];
        double[,] output = new double[numberOfOutputNeurons, 1];
        double[,] target = new double[numberOfOutputNeurons, 1];
        
        double[,] weightsIH = new double[numberOfHiddenLayer1Neurons, numberOfInputNeurons];
        double[,] deltaWeightsIH = new double[numberOfHiddenLayer1Neurons, numberOfInputNeurons];
        
        double[,] weightsHO = new double[numberOfOutputNeurons, numberOfHiddenLayer1Neurons];
        double[,] deltaWeightsHO = new double[numberOfOutputNeurons, numberOfHiddenLayer1Neurons];
        
        double[,] errorOutput = new double[numberOfOutputNeurons, numberOfHiddenLayer1Neurons];
        double[,] errorHidden1 = new double[numberOfOutputNeurons, numberOfInputNeurons];

        double[,] ones = new double[numberOfOutputNeurons, 1];

        public void Train()
        {
            Initialize();
            Calculation();
            Print();
        }
        void Initialize()
        {
            int iterator = 0;
            for (int i = 0; i < numberOfInputNeurons; i++)
            {
                inputActivation[i, 0] = GetRandomNumber(-1.00,1.00,i);
                inputActivationTranspose[0, i] = inputActivation[i, 0];
            }
            for (int i = 0; i < numberOfHiddenLayer1Neurons; i++)
            {
                for (int j = 0; j < numberOfInputNeurons; j++)
                {
                    weightsIH[i, j] = GetRandomNumber(-1.00, 1.00, iterator);
                    iterator++;
                }
            }
            iterator = 0;
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                for (int j = 0; j < numberOfHiddenLayer1Neurons; j++)
                {
                    weightsHO[i, j] = GetRandomNumber(-1.00, 1.00, iterator);
                    iterator++;
                }
            }
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                ones[i, 0] = 1;
            }

            target[0, 0] = 0;
            target[1, 0] = 1;
            //target[2, 0] = 0;
            //target[3, 0] = 0;
            //target[4, 0] = 0;
            //target[5, 0] = 0;
            //target[6, 0] = 0;
            //target[7, 0] = 0;
            //target[8, 0] = 0;
            //target[9, 0] = 0;


        }
        void Calculation()
        {
            hiddenLayer1ActivationWithoutSig = MM(weightsIH, numberOfHiddenLayer1Neurons, numberOfInputNeurons, inputActivation, 1);
            for (int i = 0; i < numberOfHiddenLayer1Neurons; i++)
            {
                hiddenLayer1Activation[i, 0] = Sig(hiddenLayer1ActivationWithoutSig[i, 0]);
                hiddenLayer1ActivationTranspose[0, i] = hiddenLayer1Activation[i, 0];
            }
            outputActivation = MM(weightsHO, numberOfOutputNeurons, numberOfHiddenLayer1Neurons, hiddenLayer1Activation, 1);
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                output[i, 0] = Sig(outputActivation[i, 0]);
                outputActivationTranspose[0, i] = output[i, 0];
            }

            //for (int i = 0; i < numberOfOutputNeurons; i++)
            //{
            //    errorOutput[i, 0] = (-(output[i, 0] - target[i, 0]) * output[i, 0] * (1 - output[i, 0]));
            //}
            errorOutput = MM(errorOutput, numberOfOutputNeurons, 1, outputActivationTranspose, numberOfOutputNeurons);
            errorOutput = MM(errorOutput, numberOfOutputNeurons, 1, MS(ones,numberOfOutputNeurons,1,output), numberOfOutputNeurons);
            deltaWeightsHO = MM(errorOutput, numberOfOutputNeurons, 1, hiddenLayer1ActivationTranspose, numberOfOutputNeurons);
            //weightsHO = MSWithLearningRateMultiplication(weightsHO, numberOfOutputNeurons, numberOfOutputNeurons, deltaWeightsHO);
            for (int i = 0; i < numberOfHiddenLayer1Neurons; i++)
            {
                errorHidden1[i, 0] = (-(errorOutput[i, 0]) * hiddenLayer1Activation[i, 0] * (1 - hiddenLayer1Activation[i, 0]));
            }
            deltaWeightsIH = MM(errorHidden1, numberOfHiddenLayer1Neurons, 1, inputActivationTranspose, numberOfInputNeurons);
            //weightsIH = MSWithLearningRateMultiplication(weightsIH, numberOfHiddenLayer1Neurons, numberOfHiddenLayer1Neurons, deltaWeightsIH);
        }
        void Print()
        {
            PrintMatrix("Weight Matrix Input To Hidden Layer 1:", 2, numberOfInputNeurons, weightsIH);
            PrintMatrixFile("WeightIH.csv", numberOfHiddenLayer1Neurons, numberOfInputNeurons, weightsIH);
            Console.WriteLine();

            PrintMatrix("Matrix Input Activation:", numberOfInputNeurons, 1, inputActivation);
            PrintMatrixFile("InputActivation.csv", numberOfInputNeurons, 1, inputActivation);
            Console.WriteLine();

            PrintMatrix("Matrix Hidden Layer 1 Activation Without Sig:", numberOfHiddenLayer1Neurons, 1, hiddenLayer1ActivationWithoutSig);
            PrintMatrixFile("HiddenLayer1ActivationWithoutSig.csv", numberOfHiddenLayer1Neurons, 1, hiddenLayer1ActivationWithoutSig);
            Console.WriteLine();

            PrintMatrix("Weight Matrix Hidden Layer 1 To Output:", numberOfOutputNeurons, numberOfHiddenLayer1Neurons, weightsHO);
            Console.WriteLine();

            PrintMatrix("Matrix Hidden Layer 1 Activation With Sig:", numberOfHiddenLayer1Neurons, 1, hiddenLayer1Activation);
            Console.WriteLine();

            PrintMatrix("Matrix Output Activation Without Sig:", numberOfOutputNeurons, 1, outputActivation);
            Console.WriteLine();

            PrintMatrix("Matrix Output Activation With Sig:", numberOfOutputNeurons, 1, output);
            Console.WriteLine();

            PrintMatrix("Matrix Error Output:", numberOfOutputNeurons, 1, errorOutput);
            Console.WriteLine();

            //PrintMatrix("Matrix Error Hidden:", numberOfOutputNeurons, 1, errorHidden1);
            //Console.WriteLine();

            //PrintMatrix("Delta Weight Matrix Hidden Layer 1 To Output:", numberOfOutputNeurons, numberOfOutputNeurons, deltaWeightsHO);
            //Console.WriteLine();

            //PrintMatrix("Delta Weight Matrix Input To Hidden Layer 1:", numberOfHiddenLayer1Neurons, numberOfHiddenLayer1Neurons, deltaWeightsIH);
            //Console.WriteLine();

            //PrintMatrix("Updated Weight Matrix Hidden Layer 1 To Output:", numberOfOutputNeurons, numberOfHiddenLayer1Neurons, weightsHO);
            //Console.WriteLine();

            //PrintMatrix("Updated Weight Matrix Input To Hidden Layer 1:", numberOfHiddenLayer1Neurons, numberOfInputNeurons, weightsIH);
            //Console.WriteLine();
        }
        void PrintMatrix(string header,int row,int column,double[,]matrix)
        {
            Console.WriteLine(header);
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    Console.Write(matrix[i, j] + " ");
                }
                Console.WriteLine("");
            }
        }
        void PrintMatrixFile(String path, int row, int column, double[,] matrix)
        {
            File.WriteAllText(path,"");
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    File.AppendAllText(path, matrix[i, j] + ",");
                }
                File.AppendAllText(path,"\n");
            }
        }
        double[,] MM(double[,] A, int Arow, int AcolumnAndBRow, double[,] B, int BColumn)
        {
            double [,] C = new double[Arow, BColumn];
            for (int i = 0; i < Arow; i++)
            {
                for (int j = 0; j < BColumn; j++)
                {
                    for (int k = 0; k < AcolumnAndBRow; k++)
                    {
                        C[i, j] += A[i, k] * B[k, j];
                    }
                }
            }
            return (C);

        }
        double[,] MSWithLearningRateMultiplication(double[,] A, int row, int column, double[,] B)
        {
            double[,] C = new double[row, column];
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    C[i, j] = (A[i, j] - B[i, j]) * learningRate;
                }
            }
            return (C);

        }
        double[,] MS(double[,] A, int row, int column, double[,] B)
        {
            double[,] C = new double[row, column];
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    C[i, j] = (A[i, j] - B[i, j]);
                }
            }
            return (C);

        }
        double Sig(double x)
        {
            double sig;
            sig = 1 / (1 + Math.Pow(Math.E, -x));
            return (sig);
        }
        double GetRandomNumber(double min,double max,int randomizer)
        {
            double output;
            Random X = new Random();
            output = X.NextDouble() * ((max-min)/min) + (randomizer*0.0001);
            return (output);
        }

    }
}
