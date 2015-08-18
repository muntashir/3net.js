var math = require('mathjs');
var neuralMath = require('./neuralMath');

neuralNetwork.prototype.predict = function (input) {
    if (input.length !== this.layer1) return false;

    //Layer 1
    this.a1 = math.concat([1], input);

    //Layer 2
    this.z2 = math.multiply(this.theta1, this.a1);
    this.a2 = math.concat([1], neuralMath.sigmoid(this.z2));

    //Layer 3
    this.z3 = math.multiply(this.theta2, this.a2);
    this.a3 = neuralMath.sigmoid(this.z3);

    return this.a3;
}

neuralNetwork.prototype.train = function (input, label, learning_rate, iters, regularization) {
    if (!iters) iters = 500;
    if (!learning_rate) learning_rate = 0.1;
    if (!regularization) regularization = 1;
    if (label.length !== this.layer3) return false;
    if (input.length !== this.layer1) return false;

    [this.theta1, this.theta2] = neuralMath.gradientDescent(this.backpropagation, this.theta1, this.theta2, this.theta1gradient, this.theta2gradient, iters, learning_rate);
    return true;
}

neuralNetwork.prototype.backpropagation = function () {
    //Forward
    this.predict(input);

    //Backwards
    var d3 = math.subtract(this.a3, label);
    var d2 = math.dotMultiply(math.multiply(math.transpose(this.theta2), d3), neuralMath.dSigmoid(math.concat([1], this.z2)));
    d2 = math.subset(d2, math.index(math.range(1, math.size(d2)[0])));

    //Get gradients
    this.theta2gradient = math.add(this.theta2gradient, neuralMath.outerProduct(d3, this.a2));
    this.theta1gradient = math.add(this.theta1gradient, neuralMath.outerProduct(d2, this.a1));

    return [this.theta1gradient, this.theta2gradient];

}

function neuralNetwork(layer1, layer2, layer3) {
    this.layer1 = layer1;
    this.layer2 = layer2;
    this.layer3 = layer3;

    this.theta1 = math.random([layer2, layer1 + 1], -1, 1);
    this.theta2 = math.random([layer3, layer2 + 1], -1, 1);

    this.theta1gradient = math.zeros([layer2, layer1 + 1]);
    this.theta2gradient = math.zeros([layer3, layer2 + 1]);

    this.a1 = math.zeros(layer1 + 1);
    this.z2 = math.zeros(layer2 + 1);
    this.a2 = math.zeros(layer2 + 1);
    this.z3 = math.zeros(layer3 + 1);
    this.a3 = math.zeros(layer3 + 1);
}

function createNet(layer1, layer2, layer3) {
    return new neuralNetwork(layer1, layer2, layer3);
}

exports.createNet = createNet;