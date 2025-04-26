const ROWS = 28;
const COLS = 28;
const grid = document.getElementById('grid');
const predictionText = document.getElementById('prediction');
const clearButton = document.getElementById('clear');
let isDrawing = false;
let model;

// Initialize grid
const cells = [];
for (let r = 0; r < ROWS; r++) {
  cells[r] = [];
  for (let c = 0; c < COLS; c++) {
    const cell = document.createElement('div');
    cell.classList.add('cell');
    grid.appendChild(cell);
    cells[r][c] = cell;

    cell.addEventListener('mousedown', () => {
      isDrawing = true;
      cell.classList.add('active');
    });

    cell.addEventListener('mouseenter', () => {
      if (isDrawing) {
        cell.classList.add('active');
      }
    });
  }
}

document.body.addEventListener('mouseup', () => {
  isDrawing = false;
  predict();
});

clearButton.addEventListener('click', () => {
  for (let row of cells) {
    for (let cell of row) {
      cell.classList.remove('active');
    }
  }
  predictionText.textContent = 'None';
});

// Load the pre-trained model
async function loadModel() {
  model = await tf.loadLayersModel('model/model.json');
  console.log('Model loaded');
}

loadModel();

// Convert grid to tensor and make prediction
function predict() {
  if (!model) return;

  // Create a 28x28 grayscale image
  const input = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      input.push(cells[r][c].classList.contains('active') ? 1 : 0);
    }
  }

  // Create tensor
  const tensor = tf.tensor(input, [1, 28, 28, 1]);
  const prediction = model.predict(tensor);
  const predictedValue = prediction.argMax(1).dataSync()[0];
  predictionText.textContent = predictedValue;
}
