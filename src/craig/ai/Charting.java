package craig.ai;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.LogAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.util.ArrayList;
import java.util.HashMap;

public class Charting 
{

    public static void plotMSE(ArrayList<HashMap<Integer, Double>> mseList, ArrayList<String> titles) 
    {
        // Prepare dataset
        XYSeriesCollection dataset = new XYSeriesCollection();

        // Create one XYSeries per output index
        int numOutputs = mseList.get(0).size();
        XYSeries[] seriesArray = new XYSeries[numOutputs];

        for (int i = 0; i < numOutputs; i++) {
            seriesArray[i] = new XYSeries(titles.get(i));
        }

        // Fill series with data: x = epoch, y = MSE
        for (int epoch = 0; epoch < mseList.size(); epoch++) 
        {
            HashMap<Integer, Double> epochData = mseList.get(epoch);
            
            for (int outputIndex = 0; outputIndex < numOutputs; outputIndex++) 
            {
                double mse = epochData.get(outputIndex);
                seriesArray[outputIndex].add(epoch, mse);
            }
        }

        // Add all series to dataset
        for (XYSeries s : seriesArray) 
        {
            dataset.addSeries(s);
        }
        
        // Create chart
        JFreeChart chart = ChartFactory.createXYLineChart(
                "MSE per Output over Epochs",
                "Epoch",
                "MSE",
                dataset
        );

        // Switch Y axis to log scale
        XYPlot plot = chart.getXYPlot();
        LogAxis logAxis = new LogAxis("MSE");
        logAxis.setBase(10);
        logAxis.setSmallestValue(1e-12); // match small value used above
        plot.setRangeAxis(logAxis);
        
        // Show chart in JFrame
        JFrame frame = new JFrame("Training Loss");
        
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        frame.add(new ChartPanel(chart));
        
        frame.pack();
        
        frame.setVisible(true);
    }
}
