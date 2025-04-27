// app.js

// Grab canvas and context
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

// Ensure an opaque white background at startup
function clearCanvas() {
  ctx.fillStyle = 'white';               // opaque white :contentReference[oaicite:4]{index=4}
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById('prediction-result').innerText = '';
}
// Clear once on load
clearCanvas();

// Set up drawing style: black strokes on white
ctx.strokeStyle = 'black';
ctx.lineWidth   = 15;
ctx.lineCap     = 'round';

let drawing = false;
canvas.addEventListener('mousedown', () => { drawing = true; });
canvas.addEventListener('mouseup',   () => { drawing = false; ctx.beginPath(); });
canvas.addEventListener('mousemove', draw);

function draw(e) {
  if (!drawing) return;
  const x = e.offsetX, y = e.offsetY;
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

// Load the pre-trained MNIST model from tfjs-mnist-workshop
async function loadModel() {
  const MODEL_URL = 
    'https://raw.githubusercontent.com/google/tfjs-mnist-workshop/master/model/model.json';
  const model = await tf.loadLayersModel(MODEL_URL);  // async load :contentReference[oaicite:5]{index=5}
  console.log('Model loaded from tfjs-mnist-workshop');
  return model;
}

// Preprocess, invert, reshape, and predict
async function predictDigit() {
  const model = await loadModel();

  // 1) Grab pixel data → [200,200,3], 2) resize → [28,28,3]
  let tensor = tf.browser.fromPixels(canvas)
    .resizeNearestNeighbor([28, 28])
    .mean(2)                  // to grayscale → [28,28] :contentReference[oaicite:6]{index=6}
    .toFloat()
    .div(tf.scalar(255));     // normalize to [0,1]

  // 3) Flatten to [1,784], 4) Invert so digit=1, bg=0
  tensor = tensor
    .reshape([1, 784])        // Dense expects 2-D [batch,784] :contentReference[oaicite:7]{index=7}
    .neg()
    .add(tf.scalar(1));

  // Predict and display
  const prediction = model.predict(tensor);
  const classId    = prediction.argMax(1).dataSync()[0];
  document.getElementById('prediction-result')
          .innerText = `Predicted digit: ${classId}`;
}
