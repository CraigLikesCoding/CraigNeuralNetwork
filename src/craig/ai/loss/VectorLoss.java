package craig.ai.loss;

public interface VectorLoss extends Loss 
{
	// This function should be implemented for the purpose of logging actual
	// loss during training.  The calling routine will pass in an array of 
	// double values.  In each list will be and expected and actual values for each
	// output Neuron in the Layer this Loss is injected into.
	double loss(double[] expected, double[] actual);
	
	// This function should be implemented for the purpose of calculations
	// during backprop.  Because this is vector based, the call will be made from the 
	// output Layer, which will iterate over the output value pairs, and produce
	// an array of doubles (where each entry in the array will represent one output neuron.
    double[] lossDerivative(double expected[], double actual[]);

}
