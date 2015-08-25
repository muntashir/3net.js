# 3net.js

A simple library for implementing 3 layer neural networks

[![NPM](https://nodei.co/npm/3net.js.png)](https://npmjs.org/package/3net.js)

#### Initialization
    var three_net = require('3net.js');  // Install with 'npm install 3net.js'
    var inputLayer = 400;
    var hiddenLayer = 25;
    var outputLayer = 10;
    var neuron = "rectifier"; // Currently sigmoid and rectifier are supported
    
    //If neuron is not specified, the default sigmoid will be used
    var net = three_net.createNet(inputLayer, hiddenLayer, outputLayer, neuron);  
    
#### Online training 
    // If options is not specified, the default values will be used.
    options = {
        "learning_rate": 0.3,   // Learning rate for gradient descent. The default is 0.3
        "regularization": 0.3,  // Regularization parameter to prevent overfitting. The default is 0.3
    };
    
    // Data and label must be an array matching the dimensions of the input layer and output layer
    var success = net.train(data, label, options);
    
    //Returns true if training was successful
    if (success) console.log("training complete");  
    
#### Training on a set 
    // If options is not specified, the default values will be used.
    options = {
        "iters": 0.3,               // Maximum amount of time stochastic gradient descent will run. The default is 10
        "learning_rate": 0.3,       // Learning rate for gradient descent. The default is 0.3
        "regularization": 0.3,      // Regularization parameter to prevent overfitting. The default is 0.3
        "change_cost": 0.00001,     // If the change in cost between iterations is less than this, the net will stop training. The default is 0.00001
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
