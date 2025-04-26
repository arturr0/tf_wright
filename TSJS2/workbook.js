const tf = require('@tensorflow/tfjs-node');
function diag(tensor1d) {
    const size = tensor1d.shape[0];
    const indices = Array.from({ length: size }, (_, i) => [i, i]);
    
    return tf.scatterND(indices, tensor1d, [size, size]);
  }
  
// tf.zeros
const tensor = tf.zeros([1]);
tensor.print();

// "numpy()" is just .arraySync() or .dataSync() in tfjs
console.log(tensor.arraySync()); 

// tf.constant
const constant = tf.scalar(100);
console.log(constant.arraySync());

// Random normal
const randNormal = tf.randomNormal([2, 2], 0.0, 1.0);

// Random uniform
const randUniform = tf.randomUniform([2, 2], 0.0, 1.0);
randUniform.print();

// tf.range and tf.linalg.diag
const t = tf.range(1, 5, 1); 
const td = diag(t);
td.print();


// tf.eye
const tu = tf.eye(4);
tu.print();

// tf.reshape
const tReshaped = t.reshape([4, 1]);

// tf.constant
const t2 = tf.tensor2d([[1, 7], [3, 2]]);
const t3 = tf.tensor2d([[1, 7], [3, 2]]);
t3.print();

// Building a manual array and matrix transpose
const rows = 3;
const cols = 4;
const myArray = [];
for (let i = 0; i < rows; i++) {
  const row = [];
  for (let j = 0; j < cols; j++) {
    const value = i * 10 + j;
    row.push(value);
  }
  myArray.push(row);
}
const t2t = tf.transpose(tf.tensor2d(myArray));
t2t.print();

// tf.multiply
const a = tf.tensor2d([[1, 2], [3, 4]]);
const b = tf.tensor1d([5, 6]);
const c = tf.mul(a, b);
c.print();

// tf.ones
const a2 = tf.ones([2]);
a2.print();

// tf.expandDims
const b2 = tf.expandDims(a2, 1);
b2.print();

// tf.squeeze
const c2 = tf.squeeze(b2);
c2.print();

// Addition (broadcasting works!)
const sum = b2.add(c2);
sum.print();

const x = tf.tensor2d([[1, 2], [3, 4]]);
const y = tf.tensor2d([[5, 6], [7, 8]]);  // 2x1 tensor

const result = tf.matMul(x, y);
result.print();
