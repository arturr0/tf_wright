const tf = require('@tensorflow/tfjs'); // NOT @tensorflow/tfjs-node

(async () => {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  // Save to local filesystem
  await model.save('file://model');
  console.log('Dummy model saved to ./model/');
})();
