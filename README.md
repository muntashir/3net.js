# 3net.js

A simple library for implementing 3 layer neural networks

[![NPM](https://nodei.co/npm/3net.js.png)](https://npmjs.org/package/3net.js)

#### Initialization
    var three_net = require('3net.js');     // Install with 'npm install 3net.js'
    var inputLayer = 400;
    var hiddenLayer = 25;
    var outputLayer = 10;
    var neuron = "rectifier";               // The activation function. Can be "rectifier", "sigmoid", or "tanh"
    
    //If neuron is not specified, the default sigmoid will be used
    var net = three_net.createNet(inputLayer, hiddenLayer, outputLayer, neuron);  
    
#### Online training 
    // If options is not specified, the default values will be used.
    options = {
        "learning_rate": 0.3,   // Learning rate for gradient descent. The default is 0.5
        "dropconnect": 0.5,     // DropConnect parameter to prevent overfitting. Must be a value between 0 and 1. It represents the chance that a weight will be randomly set to 0 during training. The default is 0
        "regularization": 0.3,  // L2 regularization parameter to prevent overfitting. The default is 0
    };
    
    // Data and label must be an array matching the dimensions of the input layer and output layer
    var success = net.train(data, label, options);
    
    //Returns true if training was successful
    if (success) console.log("training complete");  
    
#### Training on a set 
    // If options is not specified, the default values will be used.
    options = {
        "iters": 100,               // Maximum amount of time stochastic gradient descent will run. The default is 1000
        "learning_rate": 0.5,       // Learning rate for gradient descent. The default is 0.5
        "regularization": 1,        // L2 regularization parameter to prevent overfitting. The default is 0
        "dropconnect": 0.5,         // DropConnect parameter to prevent overfitting. Must be a value between 0 and 1. It represents the chance that a weight will be randomly set to 0 during training. The default is 0
        "change_cost": 0.00001,     // If the change in cross entropy cost between iterations is less than this, the net will stop training. The default is 0.00001
    };
    
    // Data and label are arrays containing the training set
    var success = net.trainSet(dataset, labels, options);
    
    //Returns true if training was successful
    if (success) console.log("training complete");  
    
#### Predicting
    net.predict(data);  // Returns an array with the output layer activations
    
#### Importing and exporting
    var savedNet = net.exportNet();                 // Exports as JSON
    var copiedNet = three_net.importNet(savedNet);  // Imports from JSON
  
#### Example: Training an XOR
    var three_net = require('3net.js');
    var net = three_net.createNet(2, 3, 1);

    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
    labels = [[0], [1], [1], [0]];

    // Uses default options since it is not specified
    net.trainSet(inputs, labels);

    console.log(net.predict([1, 1])); // Outputs 0.020773462753469724
    console.log(net.predict([1, 0])); // Outputs 0.9836636258293651

    // Output values be slightly different when you try it because of random intialization

#### An online training example can be found [here](https://github.com/muntashir/draw3net)
