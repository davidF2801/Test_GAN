package com.GAN;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

public class GAN {

    public static void main(String[] args) throws Exception {

        int latentVectorSize = 100;
        int generatorOutputSize = 28 * 28;
        int discriminatorOutputSize = 1;

        MultiLayerConfiguration generatorConfiguration = new NeuralNetConfiguration.Builder()
                .updater(Updater.ADAM)
                .list()
                .layer(new DenseLayer.Builder().nIn(latentVectorSize).nOut(128).activation(Activation.LEAKYRELU).build())
                .layer(new DenseLayer.Builder().nIn(128).nOut(256).activation(Activation.LEAKYRELU).build())
                .layer(new DenseLayer.Builder().nIn(256).nOut(512).activation(Activation.LEAKYRELU).build())
                .layer(new DenseLayer.Builder().nIn(512).nOut(generatorOutputSize).activation(Activation.TANH).build())
                .build();

        MultiLayerConfiguration discriminatorConfiguration = new NeuralNetConfiguration.Builder()
                .updater(Updater.ADAM)
                .list()
                .layer(new DenseLayer.Builder().nIn(generatorOutputSize).nOut(512).activation(Activation.LEAKYRELU).build())
                .layer(new DenseLayer.Builder().nIn(512).nOut(256).activation(Activation.LEAKYRELU).build())
                .layer(new DenseLayer.Builder().nIn(256).nOut(128).activation(Activation.LEAKYRELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(128).nOut(discriminatorOutputSize).activation(Activation.SIGMOID).build())
                .build();

        MultiLayerNetwork generator = new MultiLayerNetwork(generatorConfiguration);
        MultiLayerNetwork discriminator = new MultiLayerNetwork(discriminatorConfiguration);

        generator.init();
        discriminator.init();

        DataSetIterator mnistTrain = new MnistDataSetIterator(100, true, 12345);
        for (int i = 0; i < 10000; i++) {
            generator.fit(randomVector(), labels());
            discriminator.fit(mnistTrain.next().getFeatures(), labels());
        }

        double[] vector = randomVector();
        System.out.println(generator.output(generatorInput(vector)));
    }

    private static double[] labels() {
        return new double[] {1};
    }

    private static INDArray generatorInput(double[] vector) {
        return Nd4j.create(vector, new int[] {1, vector.length});
    }
}
