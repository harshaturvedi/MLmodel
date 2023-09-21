package layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer{
    private long SEED;
    private final double leak = 0.01;

    private double[][] _weight;
    private int _inLength;
    private int _outLength;
    private int _learningRate;

    private double[] lastZ;
    private double[] lastX;

    public FullyConnectedLayer(int _inLength, int _outLength, long SEED, int _learningRate) {
        this._inLength = _inLength;
        this._outLength = _outLength;
        this.SEED = SEED;
        this._learningRate = _learningRate;

        _weight = new double[_inLength][_outLength];
        setRandomWeights();
    }

    public double[] fullyConnectedLayerForwardPass(double[] input){
        lastX = input;

        double[] z = new double[_outLength];
        double[] out = new double[_outLength];

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                z[j] = input[i]*_weight[i][j];
            }
        }

        lastZ = z;

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                out[j] = relu(z[j]);
            }
        }

        return out;
    }

    @Override
    public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    @Override
    public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedLayerForwardPass(input);

        if(_nextLayer != null){
            return _nextLayer.getOutput(forwardPass);
        }
        else {
            return forwardPass;
        }
    }

    @Override
    public void backPropogation(double[] dLd0) {
        double[] dLdX = new double[_inLength];

        double d0dZ;
        double dZdW;
        double dLdW;
        double dZdX;

        for (int i = 0; i < _inLength; i++) {
            double dLdX_sum = 0;

            for (int j = 0; j < _outLength; j++) {
                d0dZ = derivativeRelu(lastZ[j]);
                dZdW = lastX[i];
                dLdW = dLd0[j] * d0dZ * dZdW;
                dZdX = _weight[i][j];

                _weight[i][j] -= dLdW*_learningRate;

                dLdX_sum += dLd0[j] * d0dZ * dZdX;
            }

            dLdX[i] = dLdX_sum;
        }

        if(_prevLayer !=null){
            _prevLayer.backPropogation(dLdX);
        }
    }

    @Override
    public void backPropogation(List<double[][]> dLdO) {
        double[] vector = matrixToVector(dLdO);
        backPropogation(vector);
    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return _outLength;
    }

    public void setRandomWeights(){
        Random random = new Random(SEED);

        for (int i = 0; i < _inLength; i++) {
            for (int j = 0; j < _outLength; j++) {
                _weight[i][j] = random.nextGaussian();
            }
        }
    }

    public double relu(double input){
        if(input <= 0 ){
            return 0;
        }
        else {
            return input;
        }
    }

    public double derivativeRelu(double input){
        if(input <= 0 ){
            return leak;
        }
        else {
            return 1;
        }
    }
}
