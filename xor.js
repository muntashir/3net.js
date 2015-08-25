var three_net = require('./index.js');
var net = three_net.createNet(2, 3, 1);

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
labels = [[0], [1], [1], [0]];

// Uses default options since it is not specified
net.trainSet(inputs, labels);

console.log(net.predict([1, 1])); // Outputs 0.020773462753469724
console.log(net.predict([1, 0])); // Outputs 0.9836636258293651

// Output values be slightly different when you try it because of random intialization