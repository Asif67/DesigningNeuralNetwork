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
        public void PrintMatrixToConsole(string header, int row, int column, double[,] matrix)
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
            Console.WriteLine("Press Any Key to Continue");
            Console.ReadKey();
        }
        public void PrintMatrixFile(String path, int row, int column, double[,] matrix)
        {
            File.WriteAllText(path, "");
            for (int i = 0; i < row; i++)
            {
                Console.WriteLine("Writing To File '" + path + "' Please Wait!!");
                for (int j = 0; j < column; j++)
                {
                    File.AppendAllText(path, matrix[i, j] + ",");
                }
                File.AppendAllText(path, "\n");
            }
            Console.WriteLine(path + " File Write Complete. Press Any Key to Continue");
            Console.ReadKey();
        }
    }
}
