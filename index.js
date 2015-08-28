var neuralNetwork = require('./neuralNetwork');

function createNet(layer1, layer2, layer3, neuron) {
    return new neuralNetwork(layer1, layer2, layer3, neuron);
}

function importNet(data) {
    return new neuralNetwork(data.layer1, data.layer2, data.layer3, data.neuron, [data.theta1, data.theta2]);
}

exports.createNet = createNet;
exports.importNet = importNet;