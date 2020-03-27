using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DesigningNeuralNetwork.NonLinearity
{
    class ActivationFunctions
    {
        public double Sigmoid(double x)
        {
            double sig = 1 / (1 + Math.Pow(Math.E, -x));
            return sig;
        }
    }
}
