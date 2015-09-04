var math = require('mathjs');

function getDropConnectMatrix(theta, p) {
    var dropConnectMatrix = math.random([math.size(theta)[0], math.size(theta)[1]], 0, 1);

    for (var i = 0; i < math.size(theta)[0]; i += 1) {
        for (var j = 0; j < math.size(theta)[1]; j += 1) {
            if (dropConnectMatrix[i][j] <= p) {
                dropConnectMatrix[i][j] = 0;
            } else {
                dropConnectMatrix[i][j] = 1;
            }

        }
    }

    return dropConnectMatrix;
}

function rectify(x) {
    if (!x.length) {
        x = [x];
    }

    var x2 = [];
    for (var i = 0; i < x.length; i += 1) {
        x2.push(math.max(x[i], 0));
    }
    return x2;
}

function dRectify(x) {
    if (!x.length) {
        x = [x];
    }

    var x2 = [];
    for (var i = 0; i < x.length; i += 1) {
        if (x[i] < 0) {
            x2.push(0);
        } else {
            x2.push(1);
        }
    }
    return x2;
}

function tanh(x) {
    if (!x.length) {
        x = [x];
    }

    var x2 = [];
    for (var i = 0; i < x.length; i += 1) {
        if (x[i] > 100) {
            x2.push(1);
        } else {
            x2.push(math.tanh(x[i]));
        }
    }

    return x2;
}

function dTanh(x) {
    return math.subtract(1, math.square(tanh(x)));
}

function sigmoid(x) {
    return math.dotDivide(1, math.add(1, math.exp(math.multiply(-1, x))));
}

function dSigmoid(x) {
    return math.dotMultiply(sigmoid(x), math.subtract(1, sigmoid(x)));
}

//Multiplies a column vector with a row vector because MathJS doesn't support tranposing 1 dimensional matrices
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

//Computes the cost of a single prediction and label
function getCost(pred, actual, theta, options) {
    if (pred.length !== actual.length) return false;
    var cost = 0;
    for (var i = 0; i < pred.length; i += 1) {
        if (actual[i] === 0) cost += -math.log(1 - pred[i]);
        if (actual[i] === 1) cost += -math.log(pred[i]);
    }
    if (options.regularization) {
        var regCost = 0;
        for (var i = 0; i < math.size(theta[0])[0]; i += 1) {
            for (var j = 1; j < math.size(theta[0])[1]; j += 1) {
                regCost += theta[0][i][j] ^ 2;
            }
        }
        for (var i = 0; i < math.size(theta[1])[0]; i += 1) {
            for (var j = 1; j < math.size(theta[1])[1]; j += 1) {
                regCost += theta[1][i][j] ^ 2;
            }
        }
        cost += regCost * (options.regularization / 2);
    }
    return cost;
}

exports.getDropConnectMatrix = getDropConnectMatrix;
exports.tanh = tanh;
exports.dTanh = dTanh;
exports.rectify = rectify;
exports.dRectify = dRectify;
exports.sigmoid = sigmoid;
exports.dSigmoid = dSigmoid;
exports.outerProduct = outerProduct;
exports.getCost = getCost;