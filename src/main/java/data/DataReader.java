package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader {
    private final int rows = 28;
    private final int cols = 28;

    public List<Image> readData(String path){
        List<Image> images = new ArrayList<>();

        try(BufferedReader dataReader = new BufferedReader(new FileReader(path))){
            String line;

            while((line = dataReader.readLine()) != null){
                String[] singleLine = line.split(",");

                double[][] data = new double[rows][cols];
                int label = Integer.parseInt(singleLine[0]);

                int i = 1;
                for(int row = 0; row < rows; row++){
                    for(int col = 0; col < cols; col++){
                        data[row][col] = Double.parseDouble(singleLine[i]);
                        i++;
                    }
                }

                images.add(new Image(data, label));
            }

        }catch (Exception e){
            e.printStackTrace();
        }

        return images;
    }
}