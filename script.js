let model;
let data;

// Drawing
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

// Setup canvas correctly for MNIST
canvas.width = 280;
canvas.height = 280;
canvas.style.width = '280px';
canvas.style.height = '280px';
canvas.style.imageRendering = 'pixelated';

ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);

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
  ctx.lineWidth = 20;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'white';

  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  ctx.lineTo((e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo((e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY);
}

document.getElementById('clearBtn').addEventListener('click', clearCanvas);
document.getElementById('predictBtn').addEventListener('click', predictCanvas);

function clearCanvas() {
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  document.getElementById('prediction').innerText = 'Draw a digit above';
}

// Enhanced Model Architecture
async function getModel() {
  const model = tf.sequential();
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  // First convolutional layer
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 3,
    filters: 32,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));

  model.add(tf.layers.batchNormalization());

  // Second convolutional layer
  model.add(tf.layers.conv2d({
    kernelSize: 3,
    filters: 64,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
  model.add(tf.layers.dropout({ rate: 0.25 }));

  // Third convolutional layer
  model.add(tf.layers.conv2d({
    kernelSize: 3,
    filters: 128,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
  model.add(tf.layers.dropout({ rate: 0.25 }));

  model.add(tf.layers.flatten());

  // Dense layers
  model.add(tf.layers.dense({
    units: 256,
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));

  model.add(tf.layers.dropout({ rate: 0.5 }));

  model.add(tf.layers.dense({
    units: 10,
    activation: 'softmax',
    kernelInitializer: 'glorotNormal'
  }));

  const optimizer = tf.train.adam(0.001);
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

// Enhanced Training Process
async function trainModel(model, data) {
  const BATCH_SIZE = 64;
  const TRAIN_DATA_SIZE = NUM_TRAIN_ELEMENTS;
  const TEST_DATA_SIZE = NUM_TEST_ELEMENTS;
  const EPOCHS = 15;

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
    epochs: EPOCHS,
    shuffle: true,
    callbacks: [
      tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'val_loss', 'acc', 'val_acc'],
        { height: 200, callbacks: ['onEpochEnd'] }
      ),
      tf.callbacks.earlyStopping({
        monitor: 'val_loss',
        patience: 3,
        verbose: 1
      })
    ]
  });
}

// Enhanced Prediction with Confidence Scores
async function predictCanvas() {
  if (!model) {
    console.error('Model not loaded yet');
    return;
  }

  // Preprocess the drawn image
  const smallCanvas = document.createElement('canvas');
  smallCanvas.width = 28;
  smallCanvas.height = 28;
  const smallCtx = smallCanvas.getContext('2d');
  
  // Apply preprocessing similar to MNIST dataset
  smallCtx.fillStyle = 'black';
  smallCtx.fillRect(0, 0, 28, 28);
  smallCtx.drawImage(canvas, 0, 0, 28, 28);

  // Center of mass calculation to center the digit
  const imgData = smallCtx.getImageData(0, 0, 28, 28);
  const data = imgData.data;
  
  // Convert to grayscale and find bounding box
  const grayData = new Float32Array(28 * 28);
  let sumX = 0, sumY = 0, count = 0;
  
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const idx = (y * 28 + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const avg = (r + g + b) / 3;
      grayData[y * 28 + x] = avg / 255;
      
      if (avg > 10) { // Threshold for "ink"
        sumX += x;
        sumY += y;
        count++;
      }
    }
  }
  
  // Calculate center of mass
  const centerX = count > 0 ? Math.round(sumX / count) : 14;
  const centerY = count > 0 ? Math.round(sumY / count) : 14;
  
  // Calculate translation needed to center
  const translateX = 14 - centerX;
  const translateY = 14 - centerY;
  
  // Create centered image
  const centeredCanvas = document.createElement('canvas');
  centeredCanvas.width = 28;
  centeredCanvas.height = 28;
  const centeredCtx = centeredCanvas.getContext('2d');
  
  centeredCtx.fillStyle = 'black';
  centeredCtx.fillRect(0, 0, 28, 28);
  centeredCtx.save();
  centeredCtx.translate(translateX, translateY);
  centeredCtx.drawImage(smallCanvas, 0, 0);
  centeredCtx.restore();
  
  // Get centered image data
  const centeredImgData = centeredCtx.getImageData(0, 0, 28, 28);
  const centeredData = centeredImgData.data;
  const finalGrayData = new Float32Array(28 * 28);
  
  for (let i = 0; i < centeredData.length; i += 4) {
    const r = centeredData[i];
    const g = centeredData[i + 1];
    const b = centeredData[i + 2];
    const avg = (r + g + b) / 3;
    finalGrayData[i / 4] = avg / 255;
  }
  
  // Create tensor and predict
  const input = tf.tensor4d(finalGrayData, [1, 28, 28, 1]);
  const prediction = model.predict(input);
  const pred = prediction.argMax(1);
  const result = (await pred.data())[0];
  
  // Get confidence scores
  const scores = await prediction.data();
  const confidence = Math.round(scores[result] * 100);
  
  // Display results
  document.getElementById('prediction').innerHTML = `
    Prediction: <strong>${result}</strong><br>
    Confidence: <strong>${confidence}%</strong>
  `;
  
  // Visualize top predictions
  visualizePredictions(scores);
  
  // Dispose tensors
  input.dispose();
  prediction.dispose();
  pred.dispose();
}

function visualizePredictions(scores) {
  // Remove previous visualization if it exists
  const existingVis = document.getElementById('predictions-vis');
  if (existingVis) {
    existingVis.remove();
  }

  const predictionsContainer = document.createElement('div');
  predictionsContainer.id = 'predictions-vis';
  document.getElementById('prediction').appendChild(predictionsContainer);
  
  const values = Array.from(scores).map((score, i) => {
    return { index: i, value: score };
  });
  
  // Sort by confidence
  values.sort((a, b) => b.value - a.value);
  
  // Take top 3 predictions
  const topPredictions = values.slice(0, 3);
  
  const surface = { name: 'Top Predictions', tab: 'Predictions' };
  tfvis.render.barchart(
    surface,
    { values: topPredictions },
    {
      xLabel: 'Digit',
      yLabel: 'Confidence',
      yAxisDomain: [0, 1]
    }
  );
}

// Run everything
async function run() {
  data = new MnistData();
  await data.load();

  try {
    // Try loading model from your server (models/ directory)
    model = await tf.loadLayersModel('/models/mnist-model.json');
    console.log('✅ Loaded saved model from disk.');
  } catch (error) {
    console.warn('⚠️ Could not load model, training a new one...');
    
    // Create new model
    model = await getModel();
    
    // Train new model
    await trainModel(model, data);

    // Save model to download
    await model.save('downloads://mnist-model');

    console.log('✅ Model trained and downloaded. Move it to /models/ folder manually!');
  }

  clearCanvas();
}

document.addEventListener('DOMContentLoaded', run);