async function loadData() {
  // Create a simple tensor using TensorFlow.js
  const data = tf.tensor([1, 2, 3, 4, 5]);

  // Log tensor to console to confirm it's working
  console.log('Tensor Data:', data);
  
  // Example of some tensor operations
  const squaredData = data.square();
  squaredData.print();
  
  return data;
}

loadData();  // Calling the function to ensure it's executed
