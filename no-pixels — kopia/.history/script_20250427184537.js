import { MnistData } from './data.js';

let model, data;
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

// 1) Opaque white background (clearRect → transparent black) :contentReference[oaicite:5]{index=5}
function clearCanvas() {
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  document.getElementById('prediction').innerText = 'Prediction: –';
}
clearCanvas();

// Drawing in black on white
ctx.strokeStyle = 'black';
ctx.lineWidth   = 20;   // thick MNIST-style strokes
ctx.lineCap     = 'round';

let drawing = false;
canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup',   () => { drawing = false; ctx.beginPath(); });
canvas.addEventListener('mouseout',  () => { drawing = false; ctx.beginPath(); });
canvas.addEventListener('mousemove', e => {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width/rect.width);
  const y = (e.clientY - rect.top)  * (canvas.height/rect.height);
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x, y);
});

// 2) Crop & find center-of-mass :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}
function cropAndCenter(raw) {
  const w=raw.width, h=raw.height, d=raw.data;
  let [minX,minY,maxX,maxY]=[w,h,0,0];
  for(let y=0;y<h;y++){
    for(let x=0;x<w;x++){
      if(d[(y*w+x)*4] < 250){  // non-white pixel
        minX=Math.min(minX,x); maxX=Math.max(maxX,x);
        minY=Math.min(minY,y); maxY=Math.max(maxY,y);
      }
    }
  }
  if(maxX<minX) return null;  // blank

  const cw=maxX-minX+1, ch=maxY-minY+1;
  const tmp=document.createElement('canvas');
  tmp.width=cw; tmp.height=ch;
  tmp.getContext('2d').putImageData(raw, -minX, -minY);

  // center-of-mass shift :contentReference[oaicite:8]{index=8}
  const [cx,cy] = tf.tidy(()=>{
    const img = tf.browser.fromPixels(tmp,1).toFloat();
    const coords = tf.meshgrid(tf.range(0,ch), tf.range(0,cw), false);
    const mass = img.sum();
    const yMean = img.mul(coords[0]).sum().div(mass);
    const xMean = img.mul(coords[1]).sum().div(mass);
    return [xMean.dataSync()[0], yMean.dataSync()[0]];
  });
  return {canvas: tmp, cmx: cx, cmy: cy};
}

// 3) Resize to 20×20 & pad to 28×28 :contentReference[oaicite:9]{index=9}
function resizeAndPad(tmp) {
  const buf=document.createElement('canvas');
  buf.width=buf.height=28;
  const bctx=buf.getContext('2d');
  bctx.fillStyle='white'; bctx.fillRect(0,0,28,28);
  bctx.drawImage(tmp, 0,0,tmp.width,tmp.height, 4,4,20,20);
  return buf;
}

// 4) Grayscale & normalize to [0,1], reshape to [1,28,28,1] :contentReference[oaicite:10]{index=10}
function toTensor(buf) {
  return tf.browser.fromPixels(buf,1)
    .toFloat()
    .div(tf.scalar(255))
    .reshape([1,28,28,1]);
}

// Predict pipeline
async function predictCanvas() {
  const raw = ctx.getImageData(0,0,canvas.width,canvas.height);
  const step = cropAndCenter(raw);
  if (!step) return alert('Draw a digit first!');
  const small = resizeAndPad(step.canvas);
  const input = toTensor(small);
  const pred  = model.predict(input);
  const arr   = pred.dataSync();
  const digit = arr.indexOf(Math.max(...arr));
  const conf  = Math.max(...arr);
  document.getElementById('prediction')
          .innerText = `Prediction: ${digit} (${(conf*100).toFixed(1)}%)`;
  input.dispose(); pred.dispose();
}

document.getElementById('clearBtn').onclick   = clearCanvas;
document.getElementById('predictBtn').onclick = predictCanvas;

// Build & train CNN
async function getModel() {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({inputShape:[28,28,1],kernelSize:5,filters:8,activation:'relu'}));
  m.add(tf.layers.maxPooling2d({poolSize:[2,2]}));
  m.add(tf.layers.conv2d({kernelSize:5,filters:16,activation:'relu'}));
  m.add(tf.layers.maxPooling2d({poolSize:[2,2]}));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({units:10,activation:'softmax'}));
  m.compile({optimizer:'adam',loss:'categoricalCrossentropy',metrics:['accuracy']});
  return m;
}

async function run() {
  data = new MnistData();
  await data.load();  // load MNIST :contentReference[oaicite:11]{index=11}
  model = await getModel();
  // brief training
  const [xT,yT] = tf.tidy(()=>{
    const d = data.nextTrainBatch(6000);
    return [d.xs.reshape([6000,28,28,1]), d.labels];
  });
  await model.fit(xT,yT,{batchSize:512,epochs:5,shuffle:true});
  xT.dispose(); yT.dispose();
  clearCanvas();
}
run();
