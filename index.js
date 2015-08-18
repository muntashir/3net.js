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

neuralNetwork.prototype.backpropagation = function (input, label) {
    //Forward
    this.predict(input);

    //Backwards
    var d3 = math.subtract(this.a3, label);
    var d2 = math.dotMultiply(math.multiply(math.transpose(this.theta2), d3), neuralMath.dSigmoid(math.concat([1], this.z2)));
    d2 = math.subset(d2, math.index(math.range(1, math.size(d2)[0])));

    //Get gradients
    this.theta2gradient = neuralMath.outerProduct(d3, this.a2);
    this.theta1gradient = neuralMath.outerProduct(d2, this.a1);
}

neuralNetwork.prototype.gradientDescent = function (input, label, iters, learning_rate) {
    var lastCost = neuralMath.getCost(this.a3, label);
    for (var i = 0; i < iters; i += 1) {
        this.backpropagation(input, label);

        var currentCost = neuralMath.getCost(this.a3, label);
        if (i > 0 && math.abs(currentCost - lastCost) <= 0.001) return;
        lastCost = currentCost;

        this.theta1 = math.subtract(this.theta1, math.dotMultiply(learning_rate, this.theta1gradient));
        this.theta2 = math.subtract(this.theta2, math.dotMultiply(learning_rate, this.theta2gradient));
    }
}

neuralNetwork.prototype.train = function (input, label, learning_rate, iters, regularization) {
    if (!iters) iters = 500;
    if (!learning_rate) learning_rate = 0.5;
    if (!regularization) regularization = 1;
    if (label.length !== this.layer3) return false;
    if (input.length !== this.layer1) return false;

    this.gradientDescent(input, label, iters, learning_rate);
    return true;
}

neuralNetwork.prototype.exportNet = function () {
    var data = {};
    data.layer1 = this.layer1;
    data.layer2 = this.layer2
    data.layer3 = this.layer3;
    data.theta1 = this.theta1;
    data.theta2 = this.theta2;
    return data;
}

function neuralNetwork(layer1, layer2, layer3, theta) {
    this.layer1 = layer1;
    this.layer2 = layer2;
    this.layer3 = layer3;

    if (!theta) {
        this.theta1 = math.random([layer2, layer1 + 1], -5, 5);
        this.theta2 = math.random([layer3, layer2 + 1], -5, 5);
    } else {
        this.theta1 = theta[0];
        this.theta2 = theta[1];
    }

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

function importNet(data) {
    return new neuralNetwork(data.layer1, data.layer2, data.layer3, [data.theta1, data.theta2]);
}

exports.createNet = createNet;
exports.importNet = importNet;