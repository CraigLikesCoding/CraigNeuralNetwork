package craig.ai.loss;

import craig.ai.helpers.Helper.LossMode;

public class CCELoss implements VectorLoss 
{
	// CCE is Categorical Cross Entropy.  This is helpful for one-hot vector outputs, so we'll 
	// pair this Loss implementation with an output Layer when we've got to categorize an input  
	// into a potential output.  This will be paired with Softmax and is vector based (unlike
	// earlier implemented Loss functions like BCE and MSE, which were scalar).  
	
	@Override
	public double loss(double[] expected, double[] actual) 
	{
		double sum = 0.0;
		
		// The loss function iterates over each output neuron and sums up the  
		// negative expected value times the log of the actual value.		
        for (int i = 0; i < expected.length; i++) 
        {
            sum -= expected[i] * Math.log(actual[i] + 1e-15); // small epsilon for stability
        }
        
        return sum;
	}

	@Override
	public double[] lossDerivative(double[] expected, double[] actual) 
	{
		double[] derivativeOfLoss = new double[expected.length];
		
		// The derivative of the loss with respect to each output neuron is simply
		// actual - expected.  Actual is the softmax output (which is a vector of probabilities 
		// that sum up to 1).  Expected is the target in the one-hot vector (either 1 or 0).
		for (int i = 0; i < expected.length; i++)
		{
			derivativeOfLoss[i] = actual[i] - expected[i];
		}
		
		return derivativeOfLoss;
	}
	
	@Override
	public LossMode getMode() 
	{
		return LossMode.VECTOR;
	}
}
