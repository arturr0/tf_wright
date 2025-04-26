let model;
let isDrawing = false;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Setup canvas
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 15;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

canvas.addEventListener('mousedown', () => isDrawing = true);
canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseout', () => isDrawing = false);
canvas.addEventListener('mousemove', draw);

function draw(e) {
  if (!isDrawing) return;
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
}

// Clear canvas
document.getElementById('clear').addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  document.getElementById('prediction').textContent = 'None';
});

// Predict
document.getElementById('predict').addEventListener('click', async () => {
  if (!model) {
    alert('Loading model, please wait...');
    return;
  }

  // Preprocess the canvas
  let imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  let input = tf.browser.fromPixels(imgData, 1)
    .resizeNearestNeighbor([28, 28]) // resize to MNIST input size
    .toFloat()
    .div(255.0)
    .expandDims(0); // shape: [1, 28, 28, 1]

  let prediction = model.predict(input);
  let predictedClass = prediction.argMax(1).dataSync()[0];

  document.getElementById('prediction').textContent = predictedClass;
});

async function loadModel() {
  model = await tf.loadLayersModel('https://raw.githubusercontent.com/tensorflow/tfjs-models/master/mnist/model.json');
  console.log('Model loaded!');
}

loadModel();
