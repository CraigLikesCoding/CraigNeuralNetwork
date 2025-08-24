package craig.ai.loss;

import craig.ai.helpers.Helper.LossMode;

public class BCELoss implements ScalarLoss 
{
	// BCE is Binary Cross Entropy.  This is helpful for binary classification problems, so we'll 
	// pair this Loss implementation with an output Layer when we're deciding if the input is 
	// a thing or not a thing - like a turtle or not a turtle.  But once we add additional 
	// output neurons, it'll be time to ditch BCE for MSE.
	
	@Override
	public double loss(double expected, double actual) 
	{
		// Small epsilon to avoid log(0)
        double eps = 1e-12;
        
        actual = Math.max(eps, Math.min(1 - eps, actual));
        
        return -(expected * Math.log(actual) + (1 - expected) * Math.log(1 - actual));
	}

	@Override
	public double lossDerivative(double expected, double actual) 
	{
		// Small epsilon to avoid log(0)
		double eps = 1e-12;
		
        actual = Math.max(eps, Math.min(1 - eps, actual));
        
        // Calculate the derivative of Loss with respect to the expected input:
        return (actual - expected) / (actual * (1 - actual)); 
	}

	@Override
	public LossMode getMode()
	{
		return LossMode.SCALAR;
	}
}
