var math = require('mathjs');

function sigmoid(x) {
    return math.dotDivide(1, math.add(1, math.exp(math.multiply(-1, x))));
}

function dSigmoid(x) {
    return math.dotMultiply(sigmoid(x), math.subtract(1, sigmoid(x)));
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

function getCost(pred, actual) {
    if (pred.length !== actual.length) return false;
    var cost = 0;
    for (var i = 0; i < pred.length; i += 1) {
        if (actual[i] === 0) cost += -math.log(1 - pred[i]);
        if (actual[i] === 1) cost += -math.log(pred[i]);
    }
    return cost;
}

exports.sigmoid = sigmoid;
exports.dSigmoid = dSigmoid;
exports.outerProduct = outerProduct;
exports.getCost = getCost;