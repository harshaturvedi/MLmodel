package org.example;

import data.DataReader;
import data.Image;

import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        List<Image> images = new DataReader().readData("data/mnist_test.csv");

    }
}