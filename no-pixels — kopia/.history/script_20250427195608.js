let model;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// 1) Clear with opaque white
function clearCanvas() {
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  document.getElementById('prediction').innerText = 'Prediction: –';
}
clearCanvas();

// Drawing setup
ctx.strokeStyle = 'black';
ctx.lineWidth = 20;
ctx.lineCap = 'round';

let drawing = false;
canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup', () => { drawing = false; ctx.beginPath(); });
canvas.addEventListener('mouseout', () => { drawing = false; ctx.beginPath(); });
canvas.addEventListener('mousemove', e => {
  if (!drawing) return;
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) * (canvas.width / r.width);
  const y = (e.clientY - r.top) * (canvas.height / r.height);
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
});

// Preprocessing
function preprocess(raw) {
  const w = raw.width, h = raw.height, d = raw.data;
  let [minX, minY, maxX, maxY] = [w, h, 0, 0];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (d[(y * w + x) * 4] < 250) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }
  }
  if (maxX < minX) return null;

  const cw = maxX - minX + 1, ch = maxY - minY + 1;
  const tmp = document.createElement('canvas');
  tmp.width = cw;
  tmp.height = ch;
  tmp.getContext('2d').putImageData(raw, -minX, -minY);

  const buf = document.createElement('canvas');
  buf.width = buf.height = 28;
  const bctx = buf.getContext('2d');
  bctx.fillStyle = 'white';
  bctx.fillRect(0, 0, 28, 28);
  bctx.drawImage(tmp, 0, 0, cw, ch, 4, 4, 20, 20);
  return buf;
}

// To Tensor
function toTensor(buf) {
  return tf.browser.fromPixels(buf, 1)
    .toFloat()
    .div(tf.scalar(255))
    .reshape([1, 28, 28, 1]);
}

// Predict
async function predictCanvas() {
  const raw = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const buf = preprocess(raw);
  if (!buf) return alert('Draw a digit!');
  const input = toTensor(buf);
  const pred = model.predict(input);
  const arr = pred.dataSync();
  const dgt = arr.indexOf(Math.max(...arr));
  const conf = Math.max(...arr);
  document.getElementById('prediction')
    .innerText = `Prediction: ${dgt} (${(conf * 100).toFixed(1)}%)`;
  input.dispose();
  pred.dispose();
}
document.getElementById('clearBtn').onclick = clearCanvas;
document.getElementById('predictBtn').onclick = predictCanvas;

// Build simple CNN
async function getModel() {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: 8,
    activation: 'relu'
  }));
  m.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  m.add(tf.layers.conv2d({ kernelSize: 5, filters: 16, activation: 'relu' }));
  m.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  m.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return m;
}

// Quick Fake Training (otherwise load pre-trained model)
async function fakeTrainModel() {
  // Quick dummy training so it starts without errors
  const xs = tf.randomNormal([100, 28, 28, 1]);
  const ys = tf.oneHot(tf.randomUniform([100], 0, 10, 'int32'), 10);
  await model.fit(xs, ys, { epochs: 1 });
  xs.dispose();
  ys.dispose();
}

// Main Run
async function run() {
  model = await getModel();
  await fakeTrainModel(); // you would replace this with real training later
  clearCanvas();
}
run();
