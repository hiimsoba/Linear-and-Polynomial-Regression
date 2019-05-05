const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));
const e = tf.variable(tf.scalar(Math.random()));
const f = tf.variable(tf.scalar(Math.random()));
const g = tf.variable(tf.scalar(Math.random()));
const h = tf.variable(tf.scalar(Math.random()));
const i = tf.variable(tf.scalar(Math.random()));
const j = tf.variable(tf.scalar(Math.random()));

let x_vals = [];
let y_vals = [];

const learningRate = 0.2;
const optimizer = tf.train.adam(learningRate);

function setup() {
  createCanvas(1000, 600);
}

function mouseDragged() {
  x_vals.push(map(mouseX, 0, width, -1, 1));
  y_vals.push(map(mouseY, 0, height, 1, -1));
}

function drawPoints(x_vals, y_vals) {
  for (let i = 0; i < x_vals.length; i++) {
    fill(255);
    noStroke();
    let x = map(x_vals[i], -1, 1, 0, width);
    let y = map(y_vals[i], 1, -1, 0, height);
    ellipse(x, y, 8);
  }
}

function drawCurve() {
  tf.tidy(() => {
    let xv = [];
    for (let x = -1; x <= 1.01; x += 0.01) {
      xv.push(x);
    }
    let yv = predict(tf.tensor1d(xv)).dataSync();
    stroke(255);
    strokeWeight(2);
    noFill();
    beginShape();
    for (let i = 0; i < xv.length; i++) {
      vertex(map(xv[i], -1, 1, 0, width), map(yv[i], 1, -1, 0, height));
    }
    endShape();
  });
}

function predict(x) {
  return tf.tidy(() => {
    return x.pow(tf.scalar(9)).mul(a)
      .add(x.pow(tf.scalar(8)).mul(b))
      .add(x.pow(tf.scalar(7)).mul(c))
      .add(x.pow(tf.scalar(6)).mul(d))
      .add(x.pow(tf.scalar(5)).mul(e))
      .add(x.pow(tf.scalar(4)).mul(f))
      .add(x.pow(tf.scalar(3)).mul(g))
      .add(x.pow(tf.scalar(2)).mul(h))
      .add(x.mul(i))
      .add(j);
  });
}

function loss(predictions, labels) {
  // subtract our labels (actual values) from predictions, square the results,
  // and take the mean.
  const meanSquareError = predictions.sub(labels).square().mean();
  return meanSquareError;
}

function train(xv, yv, numIterations = 1) {
  if (xv.length === 0) {
    return;
  }
  tf.tidy(() => {
    const xs = tf.tensor1d(xv);
    const ys = tf.tensor1d(yv);
    for (let iter = 0; iter < numIterations; iter++) {
      optimizer.minimize(() => {
        const predsYs = predict(xs);
        return loss(predsYs, ys);
      });
    }
  });
}

function draw() {
  background(0);
  drawPoints(x_vals, y_vals);
  train(x_vals, y_vals);
  drawCurve();
}
