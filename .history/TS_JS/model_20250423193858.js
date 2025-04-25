const tf = require('@tensorflow/tfjs-node'); // Native Node backend

(async () => {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  await model.save('file://model');
  console.log('Model saved to ./model/');
})();
