# 3net.js

A simple library for implementing 3 layer neural networks

[![NPM](https://nodei.co/npm/3net.js.png)](https://npmjs.org/package/3net.js)

#### Initialization
    var three_net = require('3net.js');  // Install with 'npm install 3net.js'
    var inputLayer = 400;
    var hiddenLayer = 25;
    var outputLayer = 10;
    var net = three_net.createNet(inputLayer, hiddenLayer, outputLayer);  
    
#### Training
    options = {
        "iters": 500,           // Max number of iterations for gradient descent. Default is 500.
        "learning_rate": 0.5,   // Learning rate for gradient descent. Default is 0.5.
        "regularization": 0.1,  // Regularization parameter to prevent overfitting. Default is 0.1.
        "error_bound": 0.001    // If the change in cost is less than this value during gradient descent, it finishes. Default is 0.001.
    };
    
    // Data and label must be an array matching the dimensions of the input layer and output layer. 
    // If options is not specified, the default values will be used. Returns true if training was successful, else returns false.
    var success = net.train(data, label, options);
    if (success) console.log("training complete");  
    
#### Predicting
    net.predict(data);  // Returns an array with the output layer activations
    
#### Importing and exporting
    var savedNet = net.exportNet();                 // Exports as JSON
    var copiedNet = three_net.importNet(savedNet);  // Imports as JSON
