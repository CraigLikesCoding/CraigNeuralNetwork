package craig.ai;

import java.util.ArrayList;
import java.util.List;

import craig.ai.helpers.Helper;

public class Neuron 
{

	private double outputValue;
    private double bias;
    private double delta;
    
    // This property will be used for batch processing
    // The sum of ∂Loss/∂Bias values, which are basically the bias’s gradient terms.
    private double accumulatedGradient;
    
    private double weightedSum;
    
    private List<Connection> incomingConnection = new ArrayList<Connection>();
    private List<Connection> outgoingConnection = new ArrayList<Connection>();
    
    private Helper helper;
    
    private boolean isBias = false;
    
    private int id;
    
    public Neuron()
    {
    	helper = Helper.getInstance();
    	
    	bias = helper.getRandomWeight();
    	
    	id = helper.getCounter();
    }

    public Neuron(boolean isBias)
    {
    	// Generally this will only be called when setting up a Bias Neuron.  If needed, a Bias Neuron
    	// can be inserted into a Layer to help shift the results up or down.  It's not connected to
    	// any input and acts like the intercept in a linear equation.
    	this();
    	
    	setBias(isBias);
    	
    	if (isBias())
    	{
    		setOutputValue(1.0);
    	}
    }
    
    // This function will look at all the input Neurons and Connections to produce a weighted sum.
    // The goal is to multiply each incoming Neuron's output by that Connection's weight and keep
    // a running sum.  After traversing them all for this Neuron, we'll add this Neuron's bias to 
    // sum and return that result.  
    public double calculateWeightedSum()
    {
    	double sum = 0.0;

    	for (Connection connection: incomingConnection)
    	{
    		sum += connection.getFrom().getOutputValue() * connection.getWeight();
    	}
    	
    	sum += getBias();

    	weightedSum = sum;

    	return weightedSum;
    }
    
    public double getWeightedSum()
    {
    	return weightedSum;
    }
    
    public void addToAccumulatedGradient()
    {
    	accumulatedGradient += getDelta();
    }
    
    public void resetAccumulatedGradient()
    {
    	accumulatedGradient = 0;
    }

    public void printNeuron()
    {
    	if (isBias())
    	{
    		System.out.print("This is a Bias Neuron\n");
    	}
    	System.out.printf("Weighted Sum: %f, Bias: %.17f, Delta: %f, Output: %f, ID: %s%n", getWeightedSum(), getBias(), getDelta(), getOutputValue(), getId());
    	System.out.printf("accumulatedGradient is %.17f%n", getAccumulatedGradient());
    	System.out.printf("# Incoming: %d, # Outgoing: %d%n", incomingConnection.size(), outgoingConnection.size());
    	for (int i = 0; i < outgoingConnection.size(); i++)
    	{
    		System.out.printf("Outgoing Connection %d, ID: %s, To Neuron ID: %s ", i, outgoingConnection.get(i).getId(), outgoingConnection.get(i).toNeuron.getId());
    		outgoingConnection.get(i).printConnection();
    	}
    }
    
    public void addIncomingConnection(Connection in)
    {
    	incomingConnection.add(in);
    }
    
    public void addOutgoingConnection(Connection out)
    {
    	outgoingConnection.add(out);
    }
    
    public double getOutputValue() 
    {
		return outputValue;
	}
    
	public void setOutputValue(double value) 
	{
		this.outputValue = value;
	}
	
	public double getBias() 
	{
		return bias;
	}
	
	public void setBias(double bias) 
	{
		this.bias = bias;
	}
	
	public double getDelta() 
	{
		return delta;
	}
	
	public void setDelta(double delta) 
	{
		this.delta = delta;
	}
	
	public List<Connection> getIncoming() 
	{
		return incomingConnection;
	}
	
	public void setIncoming(List<Connection> incoming) 
	{
		this.incomingConnection = incoming;
	}
	
	public List<Connection> getOutgoing() 
	{
		return outgoingConnection;
	}
	
	public void setOutgoing(List<Connection> outgoing) 
	{
		this.outgoingConnection = outgoing;
	}

	public boolean isBias() {
		return isBias;
	}

	public void setBias(boolean isBias) {
		this.isBias = isBias;
	}
	
	public String getId()
	{
		return "Neuron_" + id;
	}

	public double getAccumulatedGradient() {
		return accumulatedGradient;
	}

	public void setAccumulatedGradient(double accumulatedDelta) {
		this.accumulatedGradient = accumulatedDelta;
	}
}
