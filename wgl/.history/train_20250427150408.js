// train.js
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

// Load MNIST dataset
async function loadMnistData() {
    const mnist = require('mnist'); // install with `npm install mnist`
    const set = mnist.set(8000, 2000); // 8000 training, 2000 testing
    const trainingSet = set.training;
    const testSet = set.test;

    const trainX = tf.tensor2d(trainingSet.map(d => d.input), undefined, 'float32');
    const trainY = tf.tensor2d(trainingSet.map(d => d.output), undefined, 'float32');

    const testX = tf.tensor2d(testSet.map(d => d.input), undefined, 'float32');
    const testY = tf.tensor2d(testSet.map(d => d.output), undefined, 'float32');

    return {trainX, trainY, testX, testY};
}

// Create the model
function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [784], units: 128, activation: 'relu'}));
    model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
    model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy']});
    return model;
}

async function run() {
    const {trainX, trainY, testX, testY} = await loadMnistData();
    const model = createModel();

    await model.fit(trainX, trainY, {
        epochs: 20,
        validationData: [testX, testY],
        callbacks: tf.node.tensorBoard('./logs') // optional
    });

    await model.save('file://./model');
    console.log('Model trained and saved!');
}

run();
