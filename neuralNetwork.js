var math = require('mathjs');
var neuralMath = require('./neuralMath');

neuralNetwork.prototype.predict = function (input, isTraining) {
    if (input.length !== this.layer1) return false;

    //Layer 1
    this.a1 = math.concat([1], input);

    //Layer 2
    if (isTraining && this.options.dropconnect > 0) {
        this.z2 = math.multiply(math.dotMultiply(this.theta1, neuralMath.getDropConnectMatrix(this.theta1, this.options.dropconnect)), this.a1);
    } else {
        this.z2 = math.multiply(this.theta1, this.a1);
    }
    this.a2 = math.concat([1], this.activationFunction(this.z2));

    //Layer 3
    if (isTraining && this.options.dropconnect > 0) {
        this.z3 = math.multiply(math.dotMultiply(this.theta2, neuralMath.getDropConnectMatrix(this.theta2, this.options.dropconnect)), this.a2);
    } else {
        this.z3 = math.multiply(this.theta2, this.a2);
    }
    this.a3 = this.activationFunction(this.z3);

    return this.a3;
}

neuralNetwork.prototype.backpropagation = function (input, label) {
    //Forward
    this.predict(input, true);

    //Backwards
    var d3 = math.subtract(this.a3, label);
    var d2 = math.dotMultiply(math.multiply(math.transpose(this.theta2), d3), this.activationDerivative(math.concat([1], this.z2)));
    d2 = math.subset(d2, math.index(math.range(1, math.size(d2)[0])));

    //Get gradients
    this.theta2gradient = neuralMath.outerProduct(d3, this.a2);
    this.theta1gradient = neuralMath.outerProduct(d2, this.a1);

    if (this.options.regularization) {
        for (var i = 0; i < math.size(this.theta1gradient)[0]; i += 1) {
            for (var j = 1; j < math.size(this.theta1gradient)[1]; j += 1) {
                this.theta1gradient[i][j] += this.options.regularization * this.theta1gradient[i][j];
            }
        }
        for (var i = 0; i < math.size(this.theta2gradient)[0]; i += 1) {
            for (var j = 1; j < math.size(this.theta2gradient)[1]; j += 1) {
                this.theta2gradient[i][j] += this.options.regularization * this.theta2gradient[i][j];
            }
        }
    }
}

//Performs 1 iteration of gradient descent on an input and label and returns its cost
neuralNetwork.prototype.gradientDescent = function (input, label) {
    this.backpropagation(input, label);

    this.theta1 = math.subtract(this.theta1, math.dotMultiply(this.options.learning_rate, this.theta1gradient));
    this.theta2 = math.subtract(this.theta2, math.dotMultiply(this.options.learning_rate, this.theta2gradient));

    //Make arrays if length is 1
    if (!this.a3.length) this.a3 = [this.a3];
    if (!label.length) label = [label];

    return neuralMath.getCost(this.a3, label, [this.theta1, this.theta2], this.options);
}

//Trains one set at a time for online networks
neuralNetwork.prototype.train = function (input, label, options) {
    if (label.length !== this.layer3) return false;
    if (input.length !== this.layer1) return false;

    if (!options) options = {};
    if (!options.learning_rate) options.learning_rate = 0.5;
    if (!options.regularization) options.regularization = 0;
    if (!options.dropconnect) options.dropconnect = 0;
    this.options = options;

    this.gradientDescent(input, label);
    return true;
}

//Trains on a training set using stochastic gradient descent
neuralNetwork.prototype.trainSet = function (inputs, labels, options) {
    if (!options) options = {};
    if (!options.iters) options.iters = 1000;
    if (!options.learning_rate) options.learning_rate = 0.5;
    if (!options.regularization) options.regularization = 0;
    if (!options.dropconnect) options.dropconnect = 0;
    if (!options.change_cost) options.change_cost = 0.00001;
    this.options = options;

    //Performs stochastic gradient descent
    var lastCost = 0;
    for (var i = 0; i < this.options.iters; i += 1) {
        var currentCost = 0;
        for (var m = 0; m < inputs.length; m += 1) {
            currentCost += this.gradientDescent(inputs[m], labels[m]);
        }
        //If the cost is increasing or the change in cost is less than options.change_cost, finish learning
        if (i > 0 && (math.abs(currentCost - lastCost) <= this.options.error_bound || currentCost > lastCost)) return true;
        lastCost = currentCost;
    }

    return true;
}

//Exports the network as JSON
neuralNetwork.prototype.exportNet = function () {
    var data = {};
    data.layer1 = this.layer1;
    data.layer2 = this.layer2
    data.layer3 = this.layer3;
    data.theta1 = this.theta1;
    data.theta2 = this.theta2;
    data.neuron = this.neuron;
    return data;
}

function neuralNetwork(layer1, layer2, layer3, neuron, theta) {
    this.layer1 = layer1;
    this.layer2 = layer2;
    this.layer3 = layer3;

    this.neuron = neuron;

    if (neuron == "sigmoid") {
        this.activationFunction = neuralMath.sigmoid;
        this.activationDerivative = neuralMath.dSigmoid;
    } else if (neuron == "rectifier") {
        this.activationFunction = neuralMath.rectify;
        this.activationDerivative = neuralMath.dRectify;
    } else if (neuron == "tanh") {
        this.activationFunction = neuralMath.tanh;
        this.activationDerivative = neuralMath.dTanh;
    } else {
        this.neuron = "sigmoid";
        this.activationFunction = neuralMath.sigmoid;
        this.activationDerivative = neuralMath.dSigmoid;
    }

    if (!theta) {
        this.theta1 = math.random([layer2, layer1 + 1], -0.5, 0.5);
        this.theta2 = math.random([layer3, layer2 + 1], -0.6, 0.6);
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

module.exports = neuralNetwork;