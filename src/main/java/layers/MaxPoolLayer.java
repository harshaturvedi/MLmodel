package layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer{
    private int _stepSize;
    private int _windowSize;

    private int _inLength;
    private int _inRows;
    private int _inCols;

    List<int[][]> _lastMaxRow;
    List<int[][]> _lastMaxCol;

    public MaxPoolLayer(int _stepSize, int _windowSize, int _inLength, int _inRows, int _inCols) {
        this._stepSize = _stepSize;
        this._windowSize = _windowSize;
        this._inLength = _inLength;
        this._inRows = _inRows;
        this._inCols = _inCols;
    }

    public List<double[][]> maxPoolForwardPass(List<double[][]> input){
        List<double[][]> output = new ArrayList<>();
        _lastMaxRow = new ArrayList<>();
        _lastMaxCol = new ArrayList<>();

        for (int i = 0; i < input.size(); i++) {
            output.add(pool(input.get(i)));
        }

        return output;
    }

    public double[][] pool(double[][] input){
        double[][] output = new double[getOutputRows()][getOutputCols()];

        int[][] maxRow = new int[getOutputRows()][getOutputCols()];
        int[][] maxCol = new int[getOutputRows()][getOutputCols()];

        for (int i = 0; i < getOutputRows(); i += _stepSize) {
            for (int j = 0; j < getOutputCols(); j += _stepSize) {
                double max = 0.0;
                maxRow[i][j] = -1;
                maxCol[i][j] = -1;

                for (int k = 0; k < _windowSize; k++) {
                    for (int l = 0; l < _windowSize; l++) {
                        if(max<input[i+k][j+l]){
                            max = input[i+k][j+l];
                            maxRow[i][j] = i+k;
                            maxCol[i][j] = j+l;
                        }
                    }
                }

                output[i][j] = max;
            }
        }

        _lastMaxRow.add(maxRow);
        _lastMaxCol.add(maxCol);

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        List<double[][]> outputPool = maxPoolForwardPass(input);

        return _nextLayer.getOutput(outputPool);
    }

    @Override
    public double[] getOutput(double[] input) {
        List<double[][]> matrixList = vectorToMatrix(input, _inLength, _inRows, _inCols);
        return getOutput(matrixList);
    }

    @Override
    public void backPropogation(double[] dLdO) {
        List<double[][]> matrixList = vectorToMatrix(dLdO, getOutputLength(), getOutputRows(), getOutputCols());

        backPropogation(matrixList);
    }

    @Override
    public void backPropogation(List<double[][]> dLdO) {
        List<double[][]> dXdL = new ArrayList<>();

        int i = 0;
        for (double[][] array : dLdO){
            double[][] error = new double[_inRows][_inCols];

            for (int j = 0; j < getOutputRows(); j++) {
                for (int k = 0; k < getOutputCols(); k++) {
                    int max_i = _lastMaxRow.get(i)[j][k];
                    int max_j = _lastMaxCol.get(i)[j][k];

                    if(max_i != -1){
                        error[max_i][max_j] += array[j][k];
                    }
                }
            }

            dXdL.add(error);
            i++;
        }

        if(_prevLayer != null){
            _prevLayer.backPropogation(dXdL);
        }
    }

    @Override
    public int getOutputLength() {
        return _inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows - _windowSize)/_stepSize+1;
    }

    @Override
    public int getOutputCols() {
        return (_inCols - _windowSize)/_stepSize+1;
    }

    @Override
    public int getOutputElements() {
        return getOutputRows()*getOutputCols()*_inLength;
    }
}
