package craig.ai.layers;

import java.util.List;

import craig.ai.Neuron;
import craig.ai.helpers.Helper.LossMode;
import craig.ai.loss.Loss;

public interface Layer 
{	
	
	// Method for forward feeding the system.
	// The input comes from the previous Layer.  The length will be the number of nodes in the prior Layer.
	// The output will go to the next Layer.  The length will be the number of nodes in this Layer.
	public void feedForward();

	// This function will contain the actual activation implementation for whatever 
	// activation strategy the super class goes with (sigmoid, ReLU, etc.)
	public double applyActivation(double x);
	
	// This function will handle back propagation for whatever activation strategy 
	// is implemented.  This is for the output layer only.  
    public void backwardDeltaCalculateOutputLayer(double[] expectedOutput);
    
    // This function will handle back propagation for this activation strategy.
    // This is for all layers except the output layer. 
    public void backwardDeltaCalculateOtherLayers();
    
    // This function will update the weights of the Connections from the Neurons
    // of this Layer to the Neurons of the prior Layer - will be called after backprop.
    public void updateWeightsAndBiases(double learningRate);
    
    // This function will update the weights of the Connections from the Neurons
    // of this Layer to the Neurons of the prior Layer - will be called after a 
    // batch of backprops is complete.
    public void updateWeightsAndBiasesBatch(double learningRate, int actualBatchSize);
    
    
    // In batch mode, instead of immediately updating weights and biases after each training sample,
    // we accumulate their gradients across the entire batch. These accumulated values will later be
    // averaged (or summed) and applied during the weight/bias update step.
    public void accumulateGradients();

	// This is called for hidden Layers.  It takes the weighted sum (which is summed up elsewhere)
	// and multiplies that by the derivative of the activation function. 
    public double calculateHiddenDelta(double sumOfDeltas, double actualOutput);
    
	// This calculates the error and multiplies it by the derivative of the activation function.
	// Used only for output Layers.
    public double calculateOutputDelta(double expectedOutput, double actualOutput);
	
    public void setNeurons(List<Neuron> neuronsInput);
    
    public List<Neuron> getNeurons();
    
    public void addNeuron(Neuron neuron);
    
    public void printLayer();
    
    public void setLoss(Loss loss);
    
    public LossMode getLossMode();
}
