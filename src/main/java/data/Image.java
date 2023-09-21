package data;

public class Image {

    private double[][] data;
    private int label;

    public double[][] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }

    public Image(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    @Override
    public String toString(){

        StringBuilder s = new StringBuilder(label + ", \n");

        for (double[] datum : data) {
            for (int j = 0; j < data[0].length; j++) {
                s.append(datum[j]).append(", ");
            }
            s.append("\n");
        }

        return s.toString();
    }
}
