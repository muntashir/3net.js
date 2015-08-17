function sigmoid(x) {
    return math.dotDivide(1, math.add(1, math.exp(math.multiply(-1, x))));
}

function dSigmoid(x) {
    return math.dotMultiply(sigmoid(x), math.subtract(1, math.sigmoid(x)));
}

function gradientDescent(params, gradient, iters, learning_rate) {
    for (var i = 0; i < iters; i += 1) {
        params = math.subtract(params, math.dotMultiply(learning_rate, gradient));
    }
    return params;
}

exports.sigmoid = sigmoid;
exports.dSigmoid = dSigmoid;
exports.gradientDescent = gradientDescent;