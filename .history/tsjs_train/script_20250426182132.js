// import * as tf from '@tensorflow/tfjs';
import { MnistData } from './data.js';

let model;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const result = document.getElementById('result');
let drawing = false;

// Set up canvas
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Drawing functions
canvas.addEventListener('mousedown', () => { drawing = true; });
canvas.addEventListener('mouseup', () => { drawing = false; ctx.beginPath(); });
canvas.addEventListener('mousemove', draw);

function draw(event) {
  if (!drawing) return;
  ctx.lineWidth = 20;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'white';
  ctx.lineTo(event.offsetX, event.offsetY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(event.offsetX, event.offsetY);
}

// Clear canvas
document.getElementById('clear').addEventListener('click', () => {
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  result.innerText = 'Draw a digit and click Predict';
});

// Predict digit
document.getElementById('predict').addEventListener('click', async () => {
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(canvas, 0, 0, 28, 28);

  const resizedImageData = tempCtx.getImageData(0, 0, 28, 28);
  let img = tf.browser.fromPixels(resizedImageData, 1);
  img = tf.cast(img, 'float32');
  img = tf.div(tf.sub(255, img), 255);
  img = img.reshape([1, 28, 28, 1]);

  const prediction = model.predict(img);
  const probabilities = prediction.dataSync();
  const predictedDigit = probabilities.indexOf(Math.max(...probabilities));
  const confidence = Math.max(...probabilities) * 100;

  result.innerText = `Prediction: ${predictedDigit} (Confidence: ${confidence.toFixed(2)}%)`;

  img.dispose();
  prediction.dispose();
});

// Load and train model
async function run() {
  const data = new MnistData();
  await data.load();

  model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  model.add(tf.layers.flatten());

  model.add(tf.layers.dense({
    units: 10,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));

  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Start training after model is created
  await train(data);
}

async function train(data) {
  const BATCH_SIZE = 512;
  const TRAIN_BATCHES = 30;

  for (let i = 0; i < TRAIN_BATCHES; i++) {
    const batch = data.nextTrainBatch(BATCH_SIZE);
    await model.fit(batch.xs.reshape([BATCH_SIZE, 28, 28, 1]), batch.labels, {
      batchSize: BATCH_SIZE,
      epochs: 1,
    });

    tf.dispose(batch);
    console.log(`Training batch ${i + 1} / ${TRAIN_BATCHES}`);
  }

  console.log('Training complete.');
}

// Start everything
run();
