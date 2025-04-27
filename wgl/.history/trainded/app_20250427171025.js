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

// Load the pre-trained model
async function loadModel() {
    const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mnist/model.json');
    console.log("Model loaded.");
    return model;
}

// Predict the digit drawn on the canvas
async function predictDigit() {
    const model = await loadModel();

    // Preprocess the image data from the canvas
    const tensor = tf.browser.fromPixels(canvas)
        .resizeNearestNeighbor([28, 28])  // Resize to 28x28 pixels (MNIST image size)
        .mean(2)  // Convert to grayscale
        .expandDims(0)  // Add batch dimension
        .expandDims(-1)  // Add channel dimension
        .toFloat()
        .div(tf.scalar(255));  // Normalize to [0, 1]

    // Make the prediction
    const prediction = model.predict(tensor);
    const classId = prediction.argMax(1).dataSync()[0];

    // Display the predicted result
    document.getElementById("prediction-result").innerText = `Predicted digit: ${classId}`;
}
