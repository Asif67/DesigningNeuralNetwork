using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DesigningNeuralNetwork.SupervisedLearning.OneInput;
using DesigningNeuralNetwork.SupervisedLearning.MultipleInput;

namespace DesigningNeuralNetwork.SupervisedLearning
{
    class test
    {
        MultipleInput.Input X1 = new MultipleInput.Input();
        MultipleInput.Input X2 = new MultipleInput.Input();
        MultipleInput.Input X3 = new MultipleInput.Input();
        MultipleInput.Input X4 = new MultipleInput.Input();
        MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron N1HL1 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
        MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron N2HL1 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
        MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron N3HL1 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
        MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron N1HL2 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
        MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron N2HL2 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
        MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron N3HL2 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
        MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron O1 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
        MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron O2 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.Neuron();
        //MultipleInput.MultipleHiddenLayer.MultipleNeuron.MultipleOutput.Output O1 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.MultipleOutput.Output();
        //MultipleInput.MultipleHiddenLayer.MultipleNeuron.MultipleOutput.Output O2 = new MultipleInput.MultipleHiddenLayer.MultipleNeuron.MultipleOutput.Output();

        double previousActivationValue;
        double cost;
        void N1HL1F()
        {
            previousActivationValue = X1.input;
            N1HL1.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(X1.input + " * " + N1HL1.enteringCostWeight + " = " + N1HL1.weightedSum);

            previousActivationValue = X2.input;
            N1HL1.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(X2.input + " * " + N1HL1.enteringCostWeight + " = " + N1HL1.weightedSum);

            previousActivationValue = X3.input;
            N1HL1.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(X3.input + " * " + N1HL1.enteringCostWeight + " = " + N1HL1.weightedSum);

            previousActivationValue = X4.input;
            N1HL1.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(X4.input + " * " + N1HL1.enteringCostWeight + " = " + N1HL1.weightedSum);

            N1HL1.SigmoidActivationFunction();
            Console.WriteLine(X1.input + " * " + N1HL1.enteringCostWeight + " + " + X2.input + " * " + N1HL1.enteringCostWeight + " + " + X3.input + " * " + N1HL1.enteringCostWeight + " + " + X4.input + " * " + N1HL1.enteringCostWeight + " + " + N1HL1.bias + " = " + N1HL1.x);
            Console.WriteLine("1/(1+e^-" + N1HL1.x + ") = " + N1HL1.activationValue);
        }

        void N2HL1F()
        {
            previousActivationValue = X1.input;
            N2HL1.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(X1.input + " * " + N2HL1.enteringCostWeight + " = " + N2HL1.weightedSum);

            previousActivationValue = X2.input;
            N2HL1.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(X2.input + " * " + N2HL1.enteringCostWeight + " = " + N2HL1.weightedSum);

            previousActivationValue = X3.input;
            N2HL1.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(X3.input + " * " + N2HL1.enteringCostWeight + " = " + N2HL1.weightedSum);

            previousActivationValue = X4.input;
            N2HL1.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(X4.input + " * " + N2HL1.enteringCostWeight + " = " + N2HL1.weightedSum);

            N2HL1.SigmoidActivationFunction();
            Console.WriteLine(X1.input + " * " + N2HL1.enteringCostWeight + " + " + X2.input + " * " + N2HL1.enteringCostWeight + " + " + X3.input + " * " + N2HL1.enteringCostWeight + " + " + X4.input + " * " + N2HL1.enteringCostWeight + " + " + N2HL1.bias + " = " + N2HL1.x);
            Console.WriteLine("1/(1+e^-" + N2HL1.x + ") = " + N2HL1.activationValue);

        }

        void N3HL1F()
        {
            previousActivationValue = X1.input;
            N3HL1.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(X1.input + " * " + N3HL1.enteringCostWeight + " = " + N3HL1.weightedSum);

            previousActivationValue = X2.input;
            N3HL1.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(X2.input + " * " + N3HL1.enteringCostWeight + " = " + N3HL1.weightedSum);

            previousActivationValue = X3.input;
            N3HL1.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(X3.input + " * " + N3HL1.enteringCostWeight + " = " + N3HL1.weightedSum);

            previousActivationValue = X4.input;
            N3HL1.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(X4.input + " * " + N3HL1.enteringCostWeight + " = " + N3HL1.weightedSum);

            N3HL1.SigmoidActivationFunction();
            Console.WriteLine(X1.input + " * " + N3HL1.enteringCostWeight + " + " + X2.input + " * " + N3HL1.enteringCostWeight + " + " + X3.input + " * " + N3HL1.enteringCostWeight + " + " + X4.input + " * " + N3HL1.enteringCostWeight + " + " + N3HL1.bias + " = " + N3HL1.x);
            Console.WriteLine("1/(1+e^-" + N3HL1.x + ") = " + N3HL1.activationValue);

        }

        void N1HL2F()
        {
            previousActivationValue = N1HL1.activationValue;
            N1HL2.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N1HL1.activationValue + " * " + N1HL2.enteringCostWeight + " = " + N1HL2.weightedSum);

            previousActivationValue = N2HL1.activationValue;
            N1HL2.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N2HL1.activationValue + " * " + N1HL2.enteringCostWeight + " = " + N1HL2.weightedSum);

            previousActivationValue = N3HL1.activationValue;
            N1HL2.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N3HL1.activationValue + " * " + N1HL2.enteringCostWeight + " = " + N1HL2.weightedSum);

            N1HL2.SigmoidActivationFunction();
            Console.WriteLine(N3HL1.activationValue + " * " + N1HL2.enteringCostWeight + " + " + N3HL1.activationValue + " * " + N1HL2.enteringCostWeight + " + " + N3HL1.activationValue + " * " + N1HL2.enteringCostWeight + " + " + N1HL2.bias + " = " + N1HL2.x);
            Console.WriteLine("1/(1+e^-" + N1HL2.x + ") = " + N1HL2.activationValue);
        }

        void N2HL2F()
        {
            previousActivationValue = N1HL1.activationValue;
            N2HL2.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N1HL1.activationValue + " * " + N2HL2.enteringCostWeight + " = " + N2HL2.weightedSum);

            previousActivationValue = N2HL1.activationValue;
            N2HL2.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N2HL1.activationValue + " * " + N2HL2.enteringCostWeight + " = " + N2HL2.weightedSum);

            previousActivationValue = N3HL1.activationValue;
            N2HL2.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N3HL1.activationValue + " * " + N2HL2.enteringCostWeight + " = " + N2HL2.weightedSum);

            N2HL2.SigmoidActivationFunction();
            Console.WriteLine(N1HL1.activationValue + " * " + N2HL2.enteringCostWeight + " + " + N2HL1.activationValue + " * " + N2HL2.enteringCostWeight + " + " + N3HL1.activationValue + " * " + N2HL2.enteringCostWeight + " + " + N2HL2.bias + " = " + N2HL2.x);
            Console.WriteLine("1/(1+e^-" + N2HL2.x + ") = " + N2HL2.activationValue);

        }

        void N3HL2F()
        {
            previousActivationValue = N1HL1.activationValue;
            N3HL2.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N1HL1.activationValue + " * " + N3HL2.enteringCostWeight + " = " + N3HL2.weightedSum);

            previousActivationValue = N2HL1.activationValue;
            N3HL2.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N2HL1.activationValue + " * " + N3HL2.enteringCostWeight + " = " + N3HL2.weightedSum);

            previousActivationValue = N3HL1.activationValue;
            N3HL2.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N3HL1.activationValue + " * " + N3HL2.enteringCostWeight + " = " + N3HL2.weightedSum);

            N3HL2.SigmoidActivationFunction();
            Console.WriteLine(N1HL1.activationValue + " * " + N3HL2.enteringCostWeight + " + " + N2HL1.activationValue + " * " + N3HL2.enteringCostWeight + " + " + N3HL1.activationValue + " * " + N3HL2.enteringCostWeight + " + " + N3HL2.bias + " = " + N3HL2.x);
            Console.WriteLine("1/(1+e^-" + N3HL2.x + ") = " + N3HL2.activationValue);

        }
        void O1F()
        {
            previousActivationValue = N1HL2.activationValue;
            O1.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N1HL2.activationValue + " * " + O1.enteringCostWeight + " = " + O1.weightedSum);

            previousActivationValue = N2HL1.activationValue;
            O1.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N2HL2.activationValue + " * " + O1.enteringCostWeight + " = " + O1.weightedSum);

            previousActivationValue = N3HL1.activationValue;
            O1.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N3HL2.activationValue + " * " + O1.enteringCostWeight + " = " + O1.weightedSum);

            O1.SigmoidActivationFunction();
            Console.WriteLine(N1HL2.activationValue + " * " + O1.enteringCostWeight + " + " + N2HL2.activationValue + " * " + O1.enteringCostWeight + " + " + N3HL2.activationValue + " * " + O1.enteringCostWeight + " + " + O1.bias + " = " + O1.x);
            Console.WriteLine("1/(1+e^-" + O1.x + ") = " + O1.activationValue);

        }
        void O2F()
        {
            previousActivationValue = N1HL2.activationValue;
            O2.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N1HL2.activationValue + " * " + O2.enteringCostWeight + " = " + O2.weightedSum);

            previousActivationValue = N2HL1.activationValue;
            O2.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N2HL2.activationValue + " * " + O2.enteringCostWeight + " = " + O2.weightedSum);

            previousActivationValue = N3HL1.activationValue;
            O2.ActivationvalueCalculation(previousActivationValue);
            Console.WriteLine(N3HL2.activationValue + " * " + O2.enteringCostWeight + " = " + O2.weightedSum);

            O2.SigmoidActivationFunction();
            Console.WriteLine(N1HL2.activationValue + " * " + O2.enteringCostWeight + " + " + N2HL2.activationValue + " * " + O2.enteringCostWeight + " + " + N3HL2.activationValue + " * " + O2.enteringCostWeight + " + " + O2.bias + " = " + O2.x);
            Console.WriteLine("1/(1+e^-" + O2.x + ") = " + O2.activationValue);

        }

        public void Initialize()
        {
            Random Y1 = new Random();
            Random Y2 = new Random();
            Random Y3 = new Random();
            Random Y4 = new Random();
            Random Y5 = new Random();
            Random Y6 = new Random();
            Random Y7 = new Random();
            Random Y8 = new Random();
            Random Y9 = new Random();
            Random Y10 = new Random();
            Random Y11 = new Random();
            Random Y12 = new Random();
            Random Y13 = new Random();
            Random Y14 = new Random();
            Random Y15 = new Random();
            Random Y16 = new Random();
            Random Y17 = new Random();
            Random Y18 = new Random();
            Random Y19 = new Random();
            Random Y20 = new Random();


            //X1.input = 0.1;
            //X2.input = 0.1;
            //X3.input = 0.1;
            //X4.input = 0.1;
            X1.input = Y1.NextDouble();
            X2.input = Y2.NextDouble();
            X3.input = Y3.NextDouble();
            X4.input = Y4.NextDouble();

            //N1HL1.enteringCostWeight = 0.01;
            //N1HL1.bias = 0.5;
            N1HL1.enteringCostWeight = Y5.NextDouble();
            N1HL1.bias = Y6.NextDouble();

            //N2HL1.enteringCostWeight = 0.01;
            //N2HL1.bias = 0.5;
            N2HL1.enteringCostWeight = Y7.NextDouble();
            N2HL1.bias = Y8.NextDouble();

            //N3HL1.enteringCostWeight = 0.01;
            //N3HL1.bias = 0.5;
            N3HL1.enteringCostWeight = Y9.NextDouble();
            N3HL1.bias = Y10.NextDouble();

            //N1HL2.enteringCostWeight = 0.02;
            //N1HL2.bias = 0.4;
            N1HL2.enteringCostWeight = Y11.NextDouble();
            N1HL2.bias = Y12.NextDouble();

            //N2HL2.enteringCostWeight = 0.02;
            //N2HL2.bias = 0.4;
            N2HL2.enteringCostWeight = Y13.NextDouble();
            N2HL2.bias = Y14.NextDouble();

            //N3HL2.enteringCostWeight = 0.02;
            //N3HL2.bias = 0.4;
            N3HL2.enteringCostWeight = Y15.NextDouble();
            N3HL2.bias = Y16.NextDouble();

            //O1.enteringCostWeight = 0.03;
            //O1.bias = 0.3;
            O1.enteringCostWeight = Y17.NextDouble();
            O1.bias = Y18.NextDouble();
            //O1.desiredOutput = 0.1;

            //O2.enteringCostWeight = 0.03;
            //O2.bias = 0.3;
            O2.enteringCostWeight = Y19.NextDouble();
            O2.bias = Y20.NextDouble();
            //O2.desiredOutput = 0.1;
        }
        public void HiddenLayer1()
        {
            Console.WriteLine("Input to Hidden Layer 1");
            Console.WriteLine("a(1,1-3) to a(1,1):");
            N1HL1F();
            Console.WriteLine("\na(1,1-3) to a(1,2):");
            N2HL1F();
            Console.WriteLine("\na(1,1-3) to a(1,3):");
            N3HL1F();
            Console.ReadKey();
        }
        public void HiddenLayer2()
        {
            Console.WriteLine("\nHidden Layer 1 to Hidden Layer 2");
            Console.WriteLine("a(1,1-2) to a(2,1):");
            N1HL2F();
            Console.WriteLine("\na(1,1-2) to a(2,2):");
            N2HL2F();
            Console.WriteLine("\na(1,1-2) to a(2,3):");
            N3HL2F();
            Console.ReadKey();
        }
        public void Output()
        {
            Console.WriteLine("\nHidden Layer 2 to Output");
            Console.WriteLine("a(1,1-2) to a(3,1):");
            O1F();
            Console.WriteLine("\na(1,1-2) to a(3,2):");
            O2F();
            Console.ReadKey();
        }
    }
}
