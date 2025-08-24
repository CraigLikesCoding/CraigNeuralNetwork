package craig.ai.loss;

public interface ScalarLoss extends Loss 
{
	// This function should be implemented for the purpose of logging actual
	// loss during training.  The calling routine should loop over all outputs
	// and call this to retrieve the results.
	double loss(double expected, double actual);
	
	// This function should be implemented for the purpose of calculations
	// during backprop.
    double lossDerivative(double expected, double actual);

}
