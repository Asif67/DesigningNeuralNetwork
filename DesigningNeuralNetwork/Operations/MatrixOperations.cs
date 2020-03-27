using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DesigningNeuralNetwork.Operations
{
    class MatrixOperations
    {
        public double[,] MatrixMultiplication(double[,] a, int m, int n, double[,] b, int p, int q)
        {
            int i, j;
            //Console.WriteLine("Matrix a:");
            //for (i = 0; i < m; i++)
            //{
            //    for (j = 0; j < n; j++)
            //    {
            //        Console.Write(a[i, j] + " ");
            //    }
            //    Console.WriteLine();
            //}
            //Console.WriteLine("Matrix b:");
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
                double[,] c = new double[m, n];
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
        public double[,] MatrixAddition(double[,] arr1, int m, int n, double[,] arr2, int p, int q)
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
        public double[,] MatrixSubstraction(double[,] arr1, int m, int n, double[,] arr2, int p, int q)
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
        public double[,] MatrixDivition(double[,] a, int m, int n, double[,] b, int p, int q)
        {
            int i, j;
            Console.WriteLine("Matrix a:");
            for (i = 0; i < m; i++)
            {
                for (j = 0; j < n; j++)
                {
                    Console.Write(a[i, j] + " ");
                }
                Console.WriteLine();
            }
            Console.WriteLine("Matrix b:");
            for (i = 0; i < p; i++)
            {
                for (j = 0; j < q; j++)
                {
                    Console.Write(b[i, j] + " ");
                }
                Console.WriteLine();
            }
            if (n != p)
            {
                Console.WriteLine("Matrix Divition not possible");
                return (a);
            }
            else
            {
                double[,] c = new double[m, n];
                for (i = 0; i < m; i++)
                {
                    for (j = 0; j < q; j++)
                    {
                        c[i, j] = 0;
                        for (int k = 0; k < n; k++)
                        {
                            c[i, j] += a[i, k] / b[k, j];
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
    }
}
