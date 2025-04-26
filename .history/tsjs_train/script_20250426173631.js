import * as tf from '@tensorflow/tfjs';
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
  if (!model) {
    alert('Please train the model first!');
    return;
  }

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
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

// Train model
document.getElementById('train').addEventListener('click', async () => {
  const data = new MnistData();
  await data.load();

  model = getModel();

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
    const d = data.nextTestBatch(TEST
::contentReference[oaicite:11]{index=11}
 
