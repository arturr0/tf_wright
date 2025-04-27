const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const mnist = require('mnist'); // Assuming you're using the `mnist` npm package

// Load MNIST data
async function loadMnistData() {
    const mnistData = mnist.set(8000, 1756); // 8000 for training, 1756 for testing
    const trainX = tf.tensor2d(mnistData.training.map(d => d.input), [8000, 784], 'float32');
    const trainY = tf.tensor2d(mnistData.training.map(d => d.output), [8000, 10], 'float32');
    const testX = tf.tensor2d(mnistData.test.map(d => d.input), [1756, 784], 'float32');
    const testY = tf.tensor2d(mnistData.test.map(d => d.output), [1756, 10], 'float32');
    
    console.log('trainX dtype:', trainX.dtype);
    console.log('trainY dtype:', trainY.dtype);
    console.log('testX dtype:', testX.dtype);
    console.log('testY dtype:', testY.dtype);
    
    return { trainX, trainY, testX, testY };
}

async function run() {
    const { trainX, trainY, testX, testY } = await loadMnistData();

    // Define the model
    const model = tf.sequential();

    // Add layers to the model
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [784] }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Log the input data and check dtype and shape before training
    console.log('trainX shape:', trainX.shape);
    console.log('trainY shape:', trainY.shape);
    console.log('testX shape:', testX.shape);
    console.log('testY shape:', testY.shape);

    try {
        // Train the model
        await model.fit(trainX, trainY, {
            epochs: 5,
            validationData: [testX, testY],
            callbacks: tf.node.tensorBoard('./logs') // optional
        });
    } catch (error) {
        console.error('Error during training:', error);
    }
}

run();
