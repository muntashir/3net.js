# 3net.js

[![NPM](https://nodei.co/npm/3net.js.png)](https://npmjs.org/package/3net.js)

###### Example
    var three_net = require('3net.js');  //Install with 'npm install 3net.js'
    
    //Initialization
    var inputLayer = 400;
    var hiddenLayer = 25;
    var outputLayer = 10;
    var net = three_net.createNet(inputLayer, hiddenLayer, outputLayer);  
    
    //Training
    options = {"iters": 500, "learning_rate": 0.5, "regularization": 0.1};
    net.train(data, label, options);  
    
    //Predicting
    net.predict(data);  //Returns a 10 dimensional array of the output neurons activations
    
    //Importing and exporting
    var savedNet = net.exportNet(); //Exports as JSON
    var copiedNet = three_net.importNet(savedNet); //Imports as JSON

A simple library for implementing 3 layer neural networks
