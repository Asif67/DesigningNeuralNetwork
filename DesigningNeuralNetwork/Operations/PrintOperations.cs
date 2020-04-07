using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace DesigningNeuralNetwork.Operations
{
    class PrintOperations
    {
        public void PrintMatrix(string header, int row, int column, double[,] matrix)
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
        public void PrintMatrixFile(String path, int row, int column, double[,] matrix)
        {
            File.WriteAllText(path, "");
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    File.AppendAllText(path, matrix[i, j] + ",");
                }
                File.AppendAllText(path, "\n");
            }
        }
        public void BiasUpdateFileStore(int sampleNumber, int HL1NumberofNeurons, int HL2NumberofNeurons, int numberOfOutputNeurons, double[,] hiddenLayer1Bias, double[,] hiddenLayer2Bias, double[,] outputBias)
        {
            String path = "Biases/HiddenLayer1/Bias" + sampleNumber + ".csv";
            //Initialize bias Matrix
            File.WriteAllText(path, Convert.ToString(""));
            for (int i = 0; i < HL1NumberofNeurons; i++)
            {
                Random X1 = new Random();
                //hiddenLayer1Bias[i, 0] += deltaBias;
                File.AppendAllText(path, Convert.ToString(hiddenLayer1Bias[i, 0] + "\n"));
                //hiddenLayer1Bias[i, 0] = 0.001;
                //Console.WriteLine("H1B(" + 0 + "," + i + ") = " + hiddenLayer1Bias[i, 0]);
                Console.WriteLine("Hidden Layer 1 Bias(" + 0 + "," + i + ") = File Write Ok Training Sample No = " + sampleNumber);
            }
            Console.WriteLine("\n");
            path = "Biases/HiddenLayer2/Bias" + sampleNumber + ".csv";
            File.WriteAllText(path, Convert.ToString(""));
            for (int i = 0; i < HL2NumberofNeurons; i++)
            {
                Random X2 = new Random();
                //hiddenLayer2Bias[i, 0] += deltaBias;
                File.AppendAllText(path, Convert.ToString(hiddenLayer2Bias[i, 0] + "\n"));
                //hiddenLayer2Bias[i, 0] = 0.002;
                //Console.WriteLine("H2B(" + 0 + "," + i + ") = " + hiddenLayer2Bias[i, 0]);
                Console.WriteLine("Hidden Layer 2 Bias(" + 0 + "," + i + ") = File Write OK Training Sample No = " + sampleNumber);
            }
            Console.WriteLine("\n");
            path = "Biases/Output/Bias" + sampleNumber + ".csv";
            File.WriteAllText(path, Convert.ToString(""));
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                Random X3 = new Random();
                //outputBias[i, 0] += deltaBias;
                File.AppendAllText(path, Convert.ToString(outputBias[i, 0] + "\n"));
                //outputBias[i, 0] = 0.003;
                //Console.WriteLine("OB(" + 0 + "," + i + ") = " + outputBias[i, 0]);
                Console.WriteLine("Output Bias(" + 0 + "," + i + ") = File Write Ok Training Sample No = " + sampleNumber);
            }
            Console.WriteLine("\n");
            //Initialize bias Matrix
        }
        public void InputIntializeFormFile(int sampleNumber, int numberOfInputNeurons, double[,] inputActivation)
        {
            Console.WriteLine(sampleNumber);
            String path = "Inputs/InputTrain" + sampleNumber + ".csv";
            //String path = "InputTrain1.csv";
            //String path = "InputTrainOneSample.csv";
            Console.WriteLine(path);
            String[] lines;
            lines = File.ReadAllLines(path);
            for (int i = 1; i < numberOfInputNeurons; i++)
            {
                inputActivation[i, 0] = Convert.ToDouble(lines[i]);
                Console.WriteLine("File Read OK:" + lines[i] + " Training Sample No = " + sampleNumber);
                Console.WriteLine("Input Assignment OK:" + inputActivation[i, 0]);
            }
        }
        public void DesiredOutputIntializeFormFile(int sampleNumber, int numberOfOutputNeurons, double[,] desiredOutput)
        {
            Console.WriteLine(sampleNumber);
            //String path = "DesiredOutputTrainOneSample.csv";
            String path = "DesiredOutputs/DesiredOutput" + sampleNumber + ".csv";
            //String path = "DesiredOutput1.csv";
            Console.WriteLine(path);
            String[] lines;
            lines = File.ReadAllLines(path);
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                desiredOutput[i, 0] = Convert.ToDouble(lines[i]);
                Console.WriteLine("File Read OK:" + lines[i]);
                Console.WriteLine("Desired Output Assignment OK:" + desiredOutput[i, 0] + " Training Sample No = " + sampleNumber);
            }
        }
        public void WeightUpdateFileStore(String path, int row, int column, int sampleNumber, double[,] weightMatrix)
        {
            Console.WriteLine("Input to Hidden Layer 1 Weight Matrix\n\n");
            File.WriteAllText(path, Convert.ToString(""));
            //Initialize Weight Matrix
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    File.AppendAllText(path, Convert.ToString(weightMatrix[i, j] + ","));
                    Console.WriteLine("Input To Hidden Layer 1 Weight Matrix(" + i + "," + j + ") = File Write Ok Training Sample No = " + sampleNumber);
                }
                File.AppendAllText(path, Convert.ToString("\n"));
                Console.WriteLine("\n");
            }
        }
    }
}
