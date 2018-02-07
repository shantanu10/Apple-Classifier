from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import data
tf.logging.set_verbosity(tf.logging.INFO)

def act_fun(x):
  if x < 0.33:
    return 0.0
  elif x > 0.66:
    return 1.0
  else:
    return 0.5
step = np.vectorize(act_fun)
npstep = lambda x: step(x).astype(np.float)
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 3])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer 1
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=npstep)
  #Dense Layer 2
  dense2 = tf.layers.dense(inputs=dense, units=256, activation=npstep)
  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 2]
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
      "logits" : logits
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  train_data,train_labels,eval_data,eval_labels = data.load_data()
  
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="D:/Projects/Apple Classifier/savedmodel/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Mak the input_fn()
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)

  #Train the model
  print("Training the classifier.....go for coffee")
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=1000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  print("evaluating the classifier.....don't go for coffee")
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=True)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  
  #Predict the results
  print("Predict the images.... wait for the result")
  predictions = mnist_classifier.predict(input_fn = eval_input_fn,hooks = [logging_hook])
  expected = eval_labels
  res = ["Apple","not apple"]
  for pred_dict, expec in zip(predictions, expected):
      template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
      class_id = pred_dict['classes']
      probability = pred_dict['probabilities'][class_id]
      print(template.format(res[class_id],100 * probability, expec))

if __name__ == "__main__":
  tf.app.run()