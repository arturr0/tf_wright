let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let drawing = false;

// Mouse event listeners for drawing on canvas
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);

function startDrawing(event) {
    drawing = true;
    draw(event);
}

function stopDrawing() {
    drawing = false;
    ctx.beginPath();
}

function draw(event) {
    if (!drawing) return;

    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById("prediction-result").innerText = ''; // Clear previous result
}

async function loadModel() {
    const MODEL_URL =
      'https://raw.githubusercontent.com/google/tfjs-mnist-workshop/master/model/model.json';
    const model = await tf.loadLayersModel(MODEL_URL);
    console.log('Model loaded from tfjs-mnist-workshop');
    return model;
  }
  
  async function predictDigit() {
    const model = await loadModel();
  
    // Preprocess and invert to match MNIST white-on-black
    const tensor = tf.browser.fromPixels(canvas)
      .resizeNearestNeighbor([28, 28])
      .mean(2)
      .toFloat()
      .div(tf.scalar(255))
      .reshape([1, 784])
      .neg()
      .add(tf.scalar(1));
  
    const prediction = model.predict(tensor);
    const classId = prediction.argMax(1).dataSync()[0];
    document.getElementById("prediction-result")
            .innerText = `Predicted digit: ${classId}`;
  }
  