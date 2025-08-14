package craig.ai;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import craig.ai.layers.Layer;

// This class is used for exporting and importing the Network.
// After a successful learning by the Network, use this class to save all final weights and 
// biases that led to successful outputs.  This will be saved as a JSON file.  You can then use
// this class to import similar JSON files to set up and seed a Network.  
public class NetworkReaderWriter 
{
	public NetworkReaderWriter()
	{
		
	}
	
	public void saveNetwork(String filename, int[] nodesCount, List<Layer> layers) throws IOException 
	{
	    Gson gson = new GsonBuilder()
	        .setPrettyPrinting()
	        .create();

	    NetworkData netData = new NetworkData();
	    
	    netData.layout = nodesCount; // your existing array
	    
	    netData.layers = new ArrayList<>();

	    for (Layer layer : layers) 
	    {
	        LayerData layerData = new LayerData();
	        
	        layerData.neurons = new ArrayList<>();
	        
	        for (Neuron neuron : layer.getNeurons()) 
	        {
	            NeuronData nData = new NeuronData();
	        
	            nData.bias = neuron.getBias();
	            
	            nData.outgoingWeights = new ArrayList<>();
	            
	            for (Connection conn : neuron.getOutgoing()) 
	            {
	                nData.outgoingWeights.add(conn.getWeight());
	            }
	            
	            layerData.neurons.add(nData);
	        }
	        
	        netData.layers.add(layerData);
	    }

	    try (FileWriter writer = new FileWriter(filename)) 
	    {
	        gson.toJson(netData, writer);
	    }
	}
	


	public void loadNetwork(String filename, int[] nodesCount, List<Layer> layers) throws IOException 
	{
	    Gson gson = new Gson();
	    
	    try (FileReader reader = new FileReader(filename)) 
	    {
	        NetworkData netData = gson.fromJson(reader, NetworkData.class);

	        // Check layout matches
	        if (Arrays.equals(netData.layout, nodesCount)) 
	        {
	            for (int l = 0; l < layers.size(); l++) 
	            {
	                Layer layer = layers.get(l);
	                
	                for (int n = 0; n < layer.getNeurons().size(); n++) 
	                {
	                    Neuron neuron = layer.getNeurons().get(n);
	                    
	                    NeuronData nData = netData.layers.get(l).neurons.get(n);
	                    
	                    neuron.setBias(nData.bias);
	                    
	                    for (int c = 0; c < neuron.getOutgoing().size(); c++) 
	                    {
	                        neuron.getOutgoing().get(c).setWeight(nData.outgoingWeights.get(c));
	                    }
	                }
	            }
	        } 
	        else 
	        {
	            throw new IllegalArgumentException("Network layout mismatch!");
	        }
	    }
	}

}



class NetworkData 
{
    int[] layout;
    
    List<LayerData> layers;
}

class LayerData 
{
    List<NeuronData> neurons;
}

class NeuronData 
{
    double bias;
    
    List<Double> outgoingWeights;
}

