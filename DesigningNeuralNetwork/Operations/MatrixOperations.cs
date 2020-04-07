using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DesigningNeuralNetwork.Operations
{
    class MatrixOperations
    {
        public double[,] InitializeMatrix(double[,] matrix, int row, int column, double initializeValue)
        {
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    matrix[i, j] = initializeValue;
                }
            }
            return (matrix);
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
        public double[,] Ones(int row, int column)
        {
            double[,] C = new double[row, column];
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    C[i, j] = 1;
                }
            }
            return (C);

        }
        public double[,] Zeros(int row, int column)
        {
            double[,] C = new double[row, column];
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    C[i, j] = 0;
                }
            }
            return (C);

        }
        public double[,] MinusOnes(int row, int column)
        {
            double[,] C = new double[row, column];
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    C[i, j] = -1;
                }
            }
            return (C);

        }
        public double[,] MatrixTranspose(double[,] A, int row, int column)
        {
            double[,] C = new double[row, column];
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    C[i, j] = A[j, i];
                }
            }
            return (C);

        }
    }
}
