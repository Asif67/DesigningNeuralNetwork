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
    }
}
