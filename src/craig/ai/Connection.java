package craig.ai;

import craig.ai.helpers.Helper;

public class Connection 
{

	Neuron fromNeuron;
    Neuron toNeuron;
    double weight;
    double gradient;

    // This property will be used for batch processing
    // The sum of ∂Loss/∂Weight values over multiple training samples before an update (mini-batch gradient).
    double accumulatedGradient;
    
    Helper helper;
    
    private int id;
    
    public Connection()
    {
    	helper = Helper.getInstance();
    	
    	weight = helper.getRandomWeight();    	
    	
    	id = helper.getCounter();
    }
    
    public Connection(Neuron from, Neuron to)
    {
    	this();
    	
    	this.fromNeuron = from;
    	this.toNeuron = to;
    }
    
    public void addToAccumulatedGradient()
    {
    	accumulatedGradient += getGradient();
    }
    
    public void resetAccumulatedGradient()
    {
    	accumulatedGradient = 0;
    }
    
    public String getId()
    {
    	return "Connection_" + id;
    }
    
    public void printConnection()
    {
    	System.out.printf("Weight: %.17f%n", getWeight());
    }
    
    public Neuron getFrom() 
    {
		return fromNeuron;
	}
    
	public void setFrom(Neuron from) 
	{
		this.fromNeuron = from;
	}
	
	public Neuron getTo() 
	{
		return toNeuron;
	}
	
	public void setTo(Neuron to) 
	{
		this.toNeuron = to;
	}
	
	public double getWeight() 
	{
		return weight;
	}
	
	public void setWeight(double weight) 
	{
		this.weight = weight;
	}
	
	public double getGradient() 
	{
		return gradient;
	}
	
	public void setGradient(double gradient) 
	{
		this.gradient = gradient;
	}

	public double getAccumulatedGradient() {
		return accumulatedGradient;
	}

	public void setAccumulatedGradient(double accumulatedGradient) {
		this.accumulatedGradient = accumulatedGradient;
	}
}
