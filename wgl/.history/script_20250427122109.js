import { MnistData } from './data.js';

let model;
let data; // Declare the data variable here

async function getPretrainedModel() {
  const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mnist_v1/model.json');
  console.log('Pre-trained model loaded');
  return model;
}

async function predictCanvas() {
  const smallCanvas = document.createElement('canvas');
  smallCanvas.width = 28;
  smallCanvas.height = 28;
  const smallCtx = smallCanvas.getContext('2d');

  smallCtx.drawImage(canvas, 0, 0, 28, 28);

  const imgData = smallCtx.getImageData(0, 0, 28, 28);
  const centeredCanvas = centerImage(imgData);
  const centeredCtx = centeredCanvas.getContext('2d');
  const centeredImgData = centeredCtx.getImageData(0, 0, 28, 28);

  const data = centeredImgData.data;

  const grayData = new Float32Array(28 * 28);

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const avg = (r + g + b) / 3;
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

async function run() {
  data = new MnistData();  // Declare and initialize the data variable
  await data.load();

  // Load the pre-trained model
  model = await getPretrainedModel();
  clearCanvas();
}

document.addEventListener('DOMContentLoaded', run);
