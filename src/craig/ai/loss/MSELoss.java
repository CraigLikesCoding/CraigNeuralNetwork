package craig.ai.loss;

import craig.ai.helpers.Helper.LossMode;

public class MSELoss implements ScalarLoss 
{
	// MSE is Mean Squared Error.  You calculate the difference between the actual and expected 
	// output value and then square it.  You want to see the MSE going down with each epoch,
	// so track it as you iterate and even graph the results in the end.  What math fun!

	@Override
	public double loss(double expected, double actual) 
	{
		// This is the "squared" in Mean Squared Error.
		return (Math.pow(actual - expected, 2));
	}

	@Override
	public double lossDerivative(double expected, double actual) 
	{
		// The derivative of the Loss with respect to an expected value works out mathematically
		// to something as simple as actual - expected.  How nice.  This is the beauty of MSE.
		return actual - expected; 
	}
	
	@Override
	public LossMode getMode()
	{
		return LossMode.SCALAR;
	}
}
