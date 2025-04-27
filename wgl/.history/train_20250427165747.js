const tf = require('@tensorflow/tfjs-node');
const mnist = require('mnist'); // Assuming you're using the mnist npm package

// Load MNIST data
async function loadMnistData() {
    const mnistData = mnist.set(8000, 1756); // 8000 for training, 1756 for testing
    console.log('Training data length:', mnistData.training.length);
    console.log('Test data length:', mnistData.test.length);

    // Ensure the tensors are created with the 'float32' dtype explicitly
    const trainX = tf.tensor2d(mnistData.training.map(d => d.input), [mnistData.training.length, 784], 'float32');
    const trainY = tf.tensor2d(mnistData.training.map(d => d.output), [mnistData.training.length, 10], 'float32');
    const testX = tf.tensor2d(mnistData.test.map(d => d.input), [mnistData.test.length, 784], 'float32');
    const testY = tf.tensor2d(mnistData.test.map(d => d.output), [mnistData.test.length, 10], 'float32');
    
    // Log data types for debugging
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

    // Check for NaN values in the training data
    if (trainX.isNaN().any().dataSync()[0]) {
        console.log('trainX has NaN values');
    }
    if (trainY.isNaN().any().dataSync()[0]) {
        console.log('trainY has NaN values');
    }

    // Try to cast inputs to float32 explicitly and train the model
    try {
        await model.fit(trainX.cast('float32'), trainY.cast('float32'), {
            epochs: 5,
            validationData: [testX.cast('float32'), testY.cast('float32')],
            callbacks: tf.node.tensorBoard('./logs') // optional
        });
    } catch (error) {
        console.error('Error during training:', error);
    }
}

run();
