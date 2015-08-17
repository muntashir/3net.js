var math = require('mathjs');
var neuralMath = require('./neuralMath');

neuralNetwork.prototype.predict = function (input) {
    if (input.length !== this.layer1) return false;

    //Layer 1
    this.z1 = math.concat([1], input);
    this.a1 = neuralMath.sigmoid(a1);

    //Layer 2
    this.z2 = math.multiply(this.theta1, math.concat([1], z1));
    this.a2 = neuralMath.sigmoid(a2);

    //Layer 3
    this.z3 = math.multiply(this.theta2, math.concat([1], z2));
    this.a3 = neuralMath.sigmoid(a3);

    return a3;
}

neuralNetwork.prototype.train = function (input, label, learning_rate, iters) {
    if (!iters) iters = 500;
    if (!learning_rate) learning_rate = 0.1;
    if (!regularization_parameter) regularization_parameter = 0;
    if (label.length !== this.layer3) return false;

    this.predict(input);
    if (!this.a3) return false;

    var d3 = math.subtract(this.a3, label);
    var d2 = math.dotMultiply(math.multiply(math.transpose(this.theta2), d3), neuralMath.dSigmoid(this.z2));
    d2 = math.subset(d2, math.index(math.range(1, math.size(d2))));

    this.theta2gradient += math.multiply(this.d3, this.a2);
    this.theta1gradient += math.multiply(this.d2, this.a1);

    this.theta1 = neuralMath.gradientDescent(this.theta1, this.theta1gradient, iters, learning_rate);
    this.theta2 = neuralMath.gradientDescent(this.theta2, this.theta2gradient, iters, learning_rate);
    return true;
}

function neuralNetwork(layer1, layer2, layer3) {
    this.layer1 = layer1;
    this.layer2 = layer2;
    this.layer3 = layer3;

    this.theta1 = math.random([layer2, layer1 + 1]);
    this.theta2 = math.random([layer3, layer2 + 1]);

    this.theta1gradient = math.zeros([layer2, layer1 + 1]);
    this.theta2gradient = math.zeros([layer3, layer2 + 1]);

    this.z1 = math.zeros(layer1 + 1);
    this.a1 = math.zeros(layer1 + 1);
    this.z2 = math.zeros(layer2 + 1);
    this.a2 = math.zeros(layer2 + 1);
    this.z3 = math.zeros(layer3 + 1);
    this.a3 = math.zeros(layer3 + 1);
}

function createNet(layer1, layer2, layer3) {
    return new neuralNetwork(layer1, layer2, layer3);
}

module.exports = createNet;