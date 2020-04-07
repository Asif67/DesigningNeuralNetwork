using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DesigningNeuralNetwork.Operations
{
    class MatrixOperations
    {
        public double[,] TranposeMatrix(double[,] A, int m, int n)
        {
            int i, j;
            double[,] B = new double[n, m];
            //Console.Write("Enter the Order of the Matrix : ");
            //m = Convert.ToInt16(Console.ReadLine());
            //n = Convert.ToInt16(Console.ReadLine());
            //Console.Write("\nEnter The Matrix Elements : ");
            //for (i = 0; i < m; i++)
            //{
            //    for (j = 0; j < n; j++)
            //    {
            //        A[i, j] = Convert.ToInt16(Console.ReadLine());
            //    }
            //}
            //Console.Clear();
            Console.WriteLine("\nMatrix A : ");
            for (i = 0; i < m; i++)
            {
                for (j = 0; j < n; j++)
                {
                    Console.Write("{0}\t", A[i, j]);

                }
                Console.WriteLine();
            }
            Console.WriteLine("Transpose Matrix : ");

            for (i = 0; i < n; i++)
            {
                for (j = 0; j < m; j++)
                {
                    B[i, j] = A[j, i];
                    Console.Write("{0}", B[i, j]);

                }
                Console.WriteLine();
            }
            Console.Read();
            return (B);
        }
        public double[,] MatrixMultiplication(double[,] A, int Arow, int AcolumnAndBRow, double[,] B, int BColumn)
        {
            double[,] C = new double[Arow, BColumn];
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
        public double[,] MatrixSubstractionWithLearningRateMultiplication(double[,] A, int row, int column, double[,] B, double learningRate)
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
        public double[,] MatrixSubstraction(double[,] A, int row, int column, double[,] B)
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
        
    }
}
