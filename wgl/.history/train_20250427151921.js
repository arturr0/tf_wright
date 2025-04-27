const tf = require('@tensorflow/tfjs-node');
const mnist = require('mnist');  // Install with `npm install mnist`

// Load MNIST dataset
async function loadMnistData() {
    const set = mnist.set(8000, 2000); // 8000 training, 2000 testing
    const trainingSet = set.training;
    const testSet = set.test;

    // Ensure tensors are explicitly cast to 'float32'
    const trainX = tf.tensor2d(trainingSet.map(d => d.input), [8000, 784], 'float32');  // Explicit cast to float32
    const trainY = tf.tensor2d(trainingSet.map(d => d.output), [8000, 10], 'float32');  // Explicit cast to float32
    const testX = tf.tensor2d(testSet.map(d => d.input), [1756, 784], 'float32');  // Explicit cast to float32
    const testY = tf.tensor2d(testSet.map(d => d.output), [1756, 10], 'float32');  // Explicit cast to float32

    // Log the tensors' data types and shapes for debugging
    console.log('trainX dtype:', trainX.dtype);
    console.log('trainY dtype:', trainY.dtype);
    console.log('testX dtype:', testX.dtype);
    console.log('testY dtype:', testY.dtype);
    console.log('trainX shape:', trainX.shape);
    console.log('trainY shape:', trainY.shape);
    console.log('testX shape:', testX.shape);
    console.log('testY shape:', testY.shape);

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
    try {
        const {trainX, trainY, testX, testY} = await loadMnistData();

        // Log tensor shapes for debugging
        console.log('trainX shape:', trainX.shape);
        console.log('trainY shape:', trainY.shape);
        console.log('testX shape:', testX.shape);
        console.log('testY shape:', testY.shape);

        const model = createModel();

        // Train the model
        await model.fit(trainX, trainY, {
            epochs: 5,
            validationData: [testX, testY],
            callbacks: tf.node.tensorBoard('./logs') // optional
        });

        // Save the model to the filesystem
        await model.save('file://./model');
        console.log('Model trained and saved!');
    } catch (error) {
        console.error('Error during training:', error);
    }
}

run();
