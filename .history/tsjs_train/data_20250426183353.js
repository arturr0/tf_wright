// data.js
async function loadData() {
  const data = tf.tensor([1, 2, 3, 4, 5]);
  console.log('Tensor Data:', data);
  const squaredData = data.square();
  squaredData.print();
  return data;
}

loadData();  // This ensures the function is executed when data.js is loaded
