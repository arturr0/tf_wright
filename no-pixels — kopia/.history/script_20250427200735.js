// Import TensorFlow.js and tfjs-vis; also assume MnistData class is defined in data.js

import { MnistData } from './data.js';

// Set up canvas elements and UI elements (assumes these exist in HTML with given IDs).
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const processedCanvas = document.getElementById('canvas');
const processedCtx = processedCanvas.getContext('2d');
const predictBtn = document.getElementById('predict-button');
const clearBtn = document.getElementById('clear-button');
const resultDiv = document.getElementById('result');

// Initialize drawing settings (black background, white strokes).
canvas.width = 280;
canvas.height = 280;
canvas.style.backgroundColor = 'black';
ctx.strokeStyle = 'white';
ctx.lineWidth = 20;
ctx.lineCap = 'round';

// Track mouse for drawing:
let drawing = false;
canvas.addEventListener('mousedown', e => {
  drawing = true;
  ctx.beginPath();
});
canvas.addEventListener('mouseup', e => { drawing = false; });
canvas.addEventListener('mouseout', e => { drawing = false; });
canvas.addEventListener('mousemove', draw);

// For touch devices:
canvas.addEventListener('touchstart', e => {
  drawing = true;
  ctx.beginPath();
  e.preventDefault();
});
canvas.addEventListener('touchend', e => {
  drawing = false;
});
canvas.addEventListener('touchmove', e => {
  if (drawing) {
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    ctx.lineTo(x, y);
    ctx.stroke();
  }
  e.preventDefault();
});

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  ctx.lineTo(x, y);
  ctx.stroke();
}

// Clear button resets the canvas and result
clearBtn.addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'black'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  resultDiv.innerText = '';
  // Also clear processed canvas
  processedCtx.clearRect(0, 0, processedCanvas.width, processedCanvas.height);
});

// Compute bounding box, center, and return a 28x28 normalized Float32Array
function preprocessCanvas() {
  // Get image data from the canvas
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  const w = imageData.width, h = imageData.height;
  
  // Find bounding box of non-black pixels (white strokes)
  let minX = w, minY = h, maxX = 0, maxY = 0;
  for (let y = 0; y < h; ++y) {
    for (let x = 0; x < w; ++x) {
      const idx = (y * w + x) * 4;
      // Check if pixel is not black (we drew in white)
      if (data[idx] > 0 || data[idx+1] > 0 || data[idx+2] > 0) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }
  // If no drawing, return empty image
  if (maxX < minX || maxY < minY) {
    return null;
  }
  const boxWidth = maxX - minX + 1;
  const boxHeight = maxY - minY + 1;
  
  // Crop the image data to the bounding box
  const cropped = document.createElement('canvas');
  cropped.width = boxWidth;
  cropped.height = boxHeight;
  const cropCtx = cropped.getContext('2d');
  cropCtx.putImageData(imageData, -minX, -minY);
  
  // Scale digit to fit in a 20x20 box (preserve aspect ratio)
  const maxDim = Math.max(boxWidth, boxHeight);
  const scale = 20 / maxDim;
  const newW = Math.round(boxWidth * scale);
  const newH = Math.round(boxHeight * scale);
  const scaledCanvas = document.createElement('canvas');
  scaledCanvas.width = newW;
  scaledCanvas.height = newH;
  const scaledCtx = scaledCanvas.getContext('2d');
  // Draw and scale the cropped image
  scaledCtx.drawImage(cropped, 0, 0, boxWidth, boxHeight, 0, 0, newW, newH);
  
  // Compute center of mass of the scaled image (mean of white pixel coordinates)&#8203;:contentReference[oaicite:5]{index=5}
  const scaledData = scaledCtx.getImageData(0, 0, newW, newH).data;
  let sumX = 0, sumY = 0, count = 0;
  for (let y = 0; y < newH; ++y) {
    for (let x = 0; x < newW; ++x) {
      const i = (y * newW + x) * 4;
      // any non-black means part of digit (we assume white ink)
      if (scaledData[i] > 0 || scaledData[i+1] > 0 || scaledData[i+2] > 0) {
        sumX += x;
        sumY += y;
        count++;
      }
    }
  }
  // Compute average coords
  const centerX = count > 0 ? sumX / count : newW / 2;
  const centerY = count > 0 ? sumY / count : newH / 2;
  
  // Create 28x28 canvas and draw the scaled image so that center-of-mass is at (14,14)
  const finalCanvas = document.createElement('canvas');
  finalCanvas.width = 28; finalCanvas.height = 28;
  const finalCtx = finalCanvas.getContext('2d');
  finalCtx.fillStyle = 'black';
  finalCtx.fillRect(0, 0, 28, 28);
  // Compute top-left such that centerX, centerY goes to (14,14)
  const offsetX = Math.round(14 - centerX);
  const offsetY = Math.round(14 - centerY);
  finalCtx.drawImage(scaledCanvas, offsetX, offsetY);
  
  // (Optional) Display processed image on secondary canvas for visualization
  // Get image data from finalCanvas
  const finalData = finalCtx.getImageData(0, 0, 28, 28);
  // Create tensor and toPixels for display (tf.browser.toPixels expects values 0-255)
  // Convert our black(0)/white(255) image to grayscale values
  tf.browser.toPixels(tf.tensor2d(Array.from(finalData.data).filter((v,i) => i%4===0), [28, 28]), processedCanvas);
  
  // Prepare a Float32Array [1,28,28,1] normalized to [0,1]
  const floatArr = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; ++i) {
    // finalData.data is RGBA; we use the R channel (they are equal)
    floatArr[i] = finalData.data[i*4] / 255.0;
  }
  return floatArr;
}

// Build a CNN model (Conv->Conv->Pool->Conv->Pool->Dense)&#8203;:contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7}
function createModel() {
  const model = tf.sequential();
  // Input: 28x28 grayscale
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 32, kernelSize: 3, activation: 'relu', kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.conv2d({
    filters: 64, kernelSize: 3, activation: 'relu', kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
  model.add(tf.layers.conv2d({
    filters: 64, kernelSize: 3, activation: 'relu', kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
  model.add(tf.layers.flatten());
  // Add dropout for regularization
  model.add(tf.layers.dropout({ rate: 0.25 }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu', kernelInitializer: 'varianceScaling' }));
  model.add(tf.layers.dropout({ rate: 0.5 }));
  // Output layer: 10 classes
  model.add(tf.layers.dense({ units: 10, activation: 'softmax', kernelInitializer: 'varianceScaling' }));
  // Compile model
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  return model;
}

// Train the model using MNIST data, with tfjs-vis monitoring&#8203;:contentReference[oaicite:8]{index=8}
async function trainModel(model, data) {
  // Show model summary (optional)
  tfvis.show.modelSummary({ name: 'Model Architecture' }, model);
  // Prepare data tensors
  const TRAIN_SIZE = 60000;
  const TEST_SIZE = 10000;
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_SIZE);
    return [d.xs.reshape([TRAIN_SIZE, 28, 28, 1]), d.labels];
  });
  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_SIZE);
    return [d.xs.reshape([TEST_SIZE, 28, 28, 1]), d.labels];
  });
  // Set up tfjs-vis callbacks
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = { name: 'Model Training', tab: 'Model' };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics); // monitor loss/accuracy
  // Fit model: batch size and epochs
  const BATCH_SIZE = 512;
  await model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 15,            // at least 15 epochs as required
    callbacks: fitCallbacks
  });
  // Clean up
  trainXs.dispose();
  trainYs.dispose();
  testXs.dispose();
  testYs.dispose();
}

// Load data, create model, train, and then enable prediction
async function run() {
  const data = new MnistData();
  await data.load();
  const model = createModel();
  await trainModel(model, data);
  // After training, enable Predict button
  predictBtn.disabled = false;
  resultDiv.innerText = 'Draw a digit and click Predict.';
  
  // When user clicks Predict, preprocess and run model.predict
  predictBtn.addEventListener('click', async () => {
    const inputArr = preprocessCanvas();
    if (!inputArr) {
      resultDiv.innerText = 'Please draw a digit first!';
      return;
    }
    // Create a tensor [1,28,28,1] and predict
    const input = tf.tensor(inputArr, [1, 28, 28, 1]);
    const prediction = model.predict(input);
    const predClass = (await prediction.argMax(1).data())[0];
    resultDiv.innerText = `Prediction: ${predClass}`;
    input.dispose();
    prediction.dispose();
  });
}

// Start the app
run();
