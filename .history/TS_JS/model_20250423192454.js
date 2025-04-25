const tf = require('@tensorflow/tfjs'); // not tfjs-node

(async () => {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  await model.save('file://model'); // will still fail in browser! Use Node.js to run this.
})();
