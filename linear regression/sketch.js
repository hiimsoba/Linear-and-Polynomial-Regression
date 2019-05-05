let x_vals = [] ;
let y_vals = [] ;

function setup() {
  createCanvas(600, 600) ;
}

function mousePressed() {
  x_vals.push([map(mouseX, 0, width, -1, 1)]) ;
  y_vals.push([map(mouseY, 0, height, 1, -1)]) ;
}

function drawPoints(x_vals, y_vals) {
  for(let i = 0 ; i < x_vals.length ; i++) {
    fill(255) ;
    noStroke() ;
    let x = map(x_vals[i][0], -1, 1, 0, width) ;
    let y = map(y_vals[i][0], 1, -1, 0, height) ;
    ellipse(x, y, 8) ;
  }
}

function drawCurve() {
  stroke(255) ;
  strokeWeight(3) ;
  beginShape() ;
  for(let x = -1 ; x <= 1.01 ; x += 1) {
    tf.tidy(() => {
      let y = model.predict(tf.tensor2d([[x]], [1, 1])).dataSync()[0] ;
      vertex(map(x, -1, 1, 0, width), map(y, 1, -1, 0, height)) ;
    })
  }
  endShape() ;
}

function draw() {
  background(0) ;
  drawPoints(x_vals, y_vals) ;
  train(x_vals, y_vals) ;
  drawCurve() ;
}
