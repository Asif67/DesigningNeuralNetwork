using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DesigningNeuralNetwork
{
    class test
    {
        static int numberOfInputNeurons = 784;
        static int columnMatrixDef = 1;
        static int HL1NumberofNeurons = 16;
        static int HL2NumberofNeurons = 16;
        static int numberOfOutputNeurons = 10;
        public double weightInitialValue = 0.001;
        public double biasInitialValue = 0.0001;
        public double slopeInitialize = 1;
        public double interceptInitialize = 0;
        public double learningRate = 0.01;//small is better
        public double minimumStepSize = 0.001;
        public double maxCounter = 1000;

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

        public double totalDeltaBias = 0;
        public double totalDeltaWeight = 0;

        public double totalCostOverAllOutputNeurons = 0;
        public double totalCostOverAllTrainingSamples = 0;

        public int numberOfTrainingSamples = 0;

        public double derivativeOfsumOfSquaredResidualWithRespectToIntercept;
        public double derivativeOfsumOfSquaredResidualWithRespectToSlope;
        public double stepSizeIntercept;
        public double stepSizeSlope;
        public void GradientDesecent()
        {

            //cost cal
            double intercept = interceptInitialize;
            double slope = slopeInitialize;
            double observedOutput = 0;
            double input = 1;
            int counter = 0;
            //cost cal
            //update start
            while (stepSizeIntercept >= minimumStepSize || counter <= maxCounter || stepSizeSlope >= minimumStepSize)
            {
                derivativeOfsumOfSquaredResidualWithRespectToIntercept += (-2 * (observedOutput - (intercept + slope * input)));
                derivativeOfsumOfSquaredResidualWithRespectToSlope += (-2 * input * (observedOutput - (intercept + slope * input)));
                stepSizeIntercept = derivativeOfsumOfSquaredResidualWithRespectToIntercept * learningRate;
                stepSizeSlope = derivativeOfsumOfSquaredResidualWithRespectToSlope * learningRate;
                counter++;
                intercept -= stepSizeIntercept;
                slope -= stepSizeSlope;
                Console.WriteLine("Step Size Intercept =" + stepSizeIntercept + " Iter: " + counter);
                Console.WriteLine("Step Size Slope =" + stepSizeSlope);
            }
            Console.ReadKey();
            //update end
        }
    }
}
