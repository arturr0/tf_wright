import { MnistData } from './data.js';

let model;
let data;

// Drawing
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

canvas.addEventListener('mousedown', () => isDrawing = true);
canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseout', () => isDrawing = false);
canvas.addEventListener('mousemove', draw);

function draw(e) {
  if (!isDrawing) return;
  console.log("draw");
  ctx.lineWidth = 20;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'black';
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
}

document.getElementById('clearBtn').addEventListener('click', clearCanvas);
document.getElementById('predictBtn').addEventListener('click', predictCanvas);

function clearCanvas() {
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
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

  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  model.add(tf.layers.flatten());

  const NUM_OUTPUT_CLASSES = 10;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
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
  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

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
  let imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  let input = tf.browser.fromPixels(imgData, 1)
    .resizeNearestNeighbor([28, 28])
    .toFloat()
    .div(255.0)
    .expandDims(0);

  const prediction = model.predict(input);
  const pred = prediction.argMax(1);

  const result = (await pred.data())[0];

  document.getElementById('prediction').innerText = `Prediction: ${result}`;

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

  clearCanvas(); // Clear when ready
}

document.addEventListener('DOMContentLoaded', run);
