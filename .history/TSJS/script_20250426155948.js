let model;
let isDrawing = false;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const loadingText = document.getElementById('loading');

// Setup canvas
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 15;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

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
    alert('Model still loading. Please wait...');
    return;
  }

  let imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  let input = tf.browser.fromPixels(imgData, 1)
    .resizeNearestNeighbor([28, 28])
    .toFloat()
    .div(255.0)
    .expandDims(0); // shape: [1, 28, 28, 1]

  let prediction = model.predict(input);
  let predictedClass = prediction.argMax(1).dataSync()[0];

  document.getElementById('prediction').textContent = predictedClass;
});

async function loadModel() {
  model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mnist/model.json');
  console.log('Model loaded!');
  loadingText.textContent = 'Model loaded! Draw a digit!';
}

loadModel();
