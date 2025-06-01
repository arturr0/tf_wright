// MNIST Data Provider
class MnistData {
    constructor() {
      this.dataset = null;
      this.trainSize = 0;
      this.testSize = 0;
      this.trainBatchIndex = 0;
      this.testBatchIndex = 0;
    }
  
    async load() {
      const data = await Promise.all([
        fetch('https://storage.googleapis.com/learnjs-data/model-builder/mnist_images_uint8')
          .then(response => response.arrayBuffer()),
        fetch('https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8')
          .then(response => response.arrayBuffer())
      ]);
  
      const [imagesArray, labelsArray] = data;
      this.dataset = {
        images: new Uint8Array(imagesArray),
        labels: new Uint8Array(labelsArray)
      };
  
      // MNIST has 60,000 training and 10,000 test elements
      this.trainSize = 55000;
      this.testSize = 10000;
    }
  
    nextTrainBatch(batchSize) {
      return this.nextBatch(batchSize, this.trainBatchIndex, this.trainSize, () => {
        this.trainBatchIndex += batchSize;
      });
    }
  
    nextTestBatch(batchSize) {
      return this.nextBatch(batchSize, this.testBatchIndex, this.testSize, () => {
        this.testBatchIndex += batchSize;
      });
    }
  
    nextBatch(batchSize, index, size, incrementIndex) {
      if (index + batchSize > size) {
        index = 0;
      }
  
      const imagesArray = this.dataset.images.slice(index * 28 * 28, (index + batchSize) * 28 * 28);
      const labelsArray = this.dataset.labels.slice(index, index + batchSize);
  
      incrementIndex();
  
      return {
        xs: tf.tensor4d(imagesArray, [batchSize, 28, 28, 1]),
        labels: tf.oneHot(tf.tensor1d(labelsArray, 'int32'), 10).toFloat()
      };
    }
  }
  
  // Global variables for the data sizes
  const NUM_TRAIN_ELEMENTS = 55000;
  const NUM_TEST_ELEMENTS = 10000;