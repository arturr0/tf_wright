import { MnistData } from './data.js';
import { NUM_TRAIN_ELEMENTS, NUM_TEST_ELEMENTS } from './data.js';

let model;
let data;

// Drawing
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

// Setup canvas correctly for MNIST
canvas.width = 280;
canvas.height = 280;
canvas.style.width = '280px';
canvas.style.height = '280px';
canvas.style.imageRendering = 'pixelated';

ctx.fillStyle = 'black'; // 🛠 background BLACK like MNIST
ctx.fillRect(0, 0, canvas.width, canvas.height);

canvas.addEventListener('mousedown', () => isDrawing = true);
canvas.addEventListener('mouseup', () => {
  isDrawing = false;
  ctx.beginPath();
});
canvas.addEventListener('mouseout', () => {
  isDrawing = false;
  ctx.beginPath();
});
canvas.addEventListener('mousemove', draw);

function draw(e) {
  if (!isDrawing) return;
  ctx.lineWidth = 20; // 🛠 thicker lines, like MNIST digits
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'white'; // 🛠 draw white on black

  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  ctx.lineTo((e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo((e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY);
}

document.getElementById('clearBtn').addEventListener('click', clearCanvas);
document.getElementById('predictBtn').addEventListener('click', predictCanvas);

function clearCanvas() {
  ctx.fillStyle = 'black'; // 🛠 black background
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
}

// Model
async function getModel() {
  const model = tf.sequential();
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
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

  return model;
}

async function trainModel(model, data) {
  const BATCH_SIZE = 128;
  const TRAIN_DATA_SIZE = NUM_TRAIN_ELEMENTS ;
  const TEST_DATA_SIZE = NUM_TEST_ELEMENTS;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [
      d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'val_loss', 'acc', 'val_acc'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

async function predictCanvas() {
  // Shrink big canvas down to 28x28
  const smallCanvas = document.createElement('canvas');
  smallCanvas.width = 28;
  smallCanvas.height = 28;
  const smallCtx = smallCanvas.getContext('2d');

  smallCtx.drawImage(canvas, 0, 0, 28, 28);

  const imgData = smallCtx.getImageData(0, 0, 28, 28);
  const data = imgData.data;

  const grayData = new Float32Array(28 * 28);

  for (let i = 0; i < data.length; i += 4) {
    // Remember: after draw, white pixel means drawn, black means background
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const avg = (r + g + b) / 3;
    // Normalize to [0,1], background = 0, digit = 1
    grayData[i / 4] = avg / 255;
  }

  const input = tf.tensor4d(grayData, [1, 28, 28, 1]);

  const prediction = model.predict(input);
  const pred = prediction.argMax(1);
  const result = (await pred.data())[0];

  document.getElementById('prediction').innerText = `Prediction: ${result}`;

  // Dispose tensors
  input.dispose();
  prediction.dispose();
  pred.dispose();
}

// Run everything
async function run() {
  data = new MnistData();
  await data.load();
  model = await getModel();
  await trainModel(model, data);

  clearCanvas();
}

document.addEventListener('DOMContentLoaded', run);
