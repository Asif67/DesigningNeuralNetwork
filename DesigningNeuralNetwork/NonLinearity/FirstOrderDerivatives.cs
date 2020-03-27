using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DesigningNeuralNetwork.NonLinearity
{
    class FirstOrderDerivatives
    {
        public double SigmoidActivationFunction(double x)
        {
            double output;
            output = x * (1 - x);
            return output;
        }
    }
}
