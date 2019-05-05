const model = tf.sequential({
  layers : [
    tf.layers.dense({
      units: 1,
      inputShape: [1]
    })
  ]
});

model.compile({
  loss: tf.losses.meanSquaredError,
  optimizer: tf.train.adam(0.1)
});

async function train(x, y, trainingEpochs = 1) {
  const xs = tf.tensor2d(x, [x.length, 1]) ;
  const ys = tf.tensor2d(y, [y.length, 1]) ;
  await model.fit(xs, ys, {
    epochs: trainingEpochs
  });
  xs.dispose() ;
  ys.dispose() ;
}
