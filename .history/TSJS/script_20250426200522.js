import {MnistData} from './data.js';

let model;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear');
const predictBtn = document.getElementById('predict');
const result = document.getElementById('result');

ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.strokeStyle = 'white';
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height); // Fill background black

let drawing = false;

canvas.addEventListener('mousedown', (e) => {
  drawing = true;
  ctx.beginPath();
});

canvas.addEventListener('mousemove', (e) => {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
  ctx.stroke();
});

canvas.addEventListener('mouseup', () => {
  drawing = false;
});

canvas.addEventListener('touchstart', (e) => {
  e.preventDefault();
  drawing = true;
  ctx.beginPath();
});

canvas.addEventListener('touchmove', (e) => {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const touch = e.touches[0];
  ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top);
  ctx.stroke();
});

canvas.addEventListener('touchend', () => {
  drawing = false;
});

clearBtn.addEventListener('click', () => {
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
});

predictBtn.addEventListener('click', async () => {
  const tensor = preprocessCanvas(canvas);
  const prediction = model.predict(tensor);
  const pred = prediction.argMax(-1);
  const classId = (await pred.data())[0];
  result.innerText = `Prediction: ${classId}`;
  tensor.dispose();
  prediction.dispose();
  pred.dispose();
});

function preprocessCanvas(canvas) {
  let tensor = tf.browser.fromPixels(canvas, 1) // 1 for grayscale
    .resizeNearestNeighbor([28, 28]) // resize to 28x28
    .toFloat()
    .div(255.0) // normalize to [0,1]
    .expandDims(0); // make batch shape [1, 28, 28, 1]
  return tensor;
}

async function run() {
  const data = new MnistData();
  await data.load();

  model = getModel(); // use your getModel()
  await train(model, data); // use your train()

  // optional: save the trained model to IndexedDB
  // await model.save('indexeddb://my-mnist-model');
}

document.addEventListener('DOMContentLoaded', run);
