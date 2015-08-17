var math = require('mathjs');

function sigmoid(x) {
    return math.dotDivide(1, math.add(1, math.exp(math.multiply(-1, x))));
}

function dSigmoid(x) {
    return math.dotMultiply(sigmoid(x), math.subtract(1, sigmoid(x)));
}

function gradientDescent(params, gradient, iters, learning_rate) {
    for (var i = 0; i < iters; i += 1) {
        params = math.subtract(params, math.dotMultiply(learning_rate, gradient));
    }
    return params;
}

function outerProduct(x, y) {
    var z = math.zeros([math.size(x)[0], math.size(y)[0]]);
    for (var i = 0; i < math.size(x)[0]; i += 1) {
        for (var j = 0; j < math.size(y)[0]; j += 1) {
            var r = x[i] * y[j];
            z[i][j] = r;
        }
    }
    return z;
}

exports.sigmoid = sigmoid;
exports.dSigmoid = dSigmoid;
exports.gradientDescent = gradientDescent;
exports.outerProduct = outerProduct;