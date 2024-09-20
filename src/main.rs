#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(non_snake_case)]
#![allow(dead_code)]
use multicalc::numerical_derivative::derivator;
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};
use std::f64::{self, consts::E};

struct Math;

impl Math {
    fn exp(x: f64) -> f64 {
        f64::powf(E, x)
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + Self::exp(-x))
    }

    fn sigmoid_prime(x: f64) -> f64 {
        let sig = Self::sigmoid(x);
        sig * (1.0 - sig)
    }

    fn ReLU(x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    fn ReLUPrime(x: f64) -> f64 {
        if x >= 0.0 {
            1.0
        } else {
            0.0
        }
    }

    fn leaky_ReLU(x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.01 * x
        }
    }

    fn leaky_ReLU_prime(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.01
        }
    }

    fn tanh(x: f64) -> f64 {
        (Self::exp(x) - Self::exp(-x)) / (Self::exp(x) + Self::exp(-x))
    }

    fn tanh_prime(x: f64) -> f64 {
        1.0 - f64::powf(Self::tanh(x), 2.0)
    }

    fn dot_product(v1: &[f64], v2: &[f64]) -> f64 {
        v1.iter().zip(v2.iter()).map(|(x, y)| x * y).sum()
    }

    fn mult_scalar_and_vector(v: &[f64], s: f64) -> Vec<f64> {
        v.iter().map(|&x| s * x).collect()
    }

    fn transpose(v: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if v.is_empty() || v[0].is_empty() {
            return vec![];
        }

        let rows = v.len();
        let cols = v[0].len();
        let mut res = vec![vec![0.0; rows]; cols];

        for i in 0..rows {
            for j in 0..cols {
                res[j][i] = v[i][j];
            }
        }
        res
    }
}

struct NeuralNetwork {
    pub biases: Vec<f64>,
    pub weights: Vec<Vec<f64>>,
}

impl NeuralNetwork {
    pub fn new(num_layers: usize, layer_sizes: &[usize]) -> Self {
        let mut nn = NeuralNetwork {
            biases: Vec::new(),
            weights: Vec::new(),
        };

        // Initialize biases and weights
        for size in layer_sizes {
            nn.bias_init(size);
            nn.weight_init(size, &layer_sizes);
        }

        nn
    }

    fn bias_init(&mut self, size: &usize) {
        let mut rng = rand::thread_rng();
        for _ in 0..*size {
            self.biases.push(rng.gen_range(-1.0..1.0));
        }
    }

    fn weight_init(&mut self, size: &usize, layer_sizes: &[usize]) {
        let mut rng = rand::thread_rng();
        let last_size = if layer_sizes.len() > 1 {
            layer_sizes[layer_sizes.len() - 2]
        } else {
            *size
        };
        let mut layer_weights = Vec::new();

        for _ in 0..*size {
            let mut weights: Vec<f64> = (0..last_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
            layer_weights.append(&mut weights);
        }
        self.weights.push(layer_weights);
    }

    pub fn feedforward(&self, input: Vec<f64>) -> Vec<f64> {
        let mut output = input;

        for (i, weights) in self.weights.iter().enumerate() {
            let z = Math::dot_product(&output, weights); // Simple linear combination
            let activated = Math::sigmoid(z + self.biases[i]); // Activation function
            output = vec![activated];
        }

        output
    }
}

trait Layer {
    // pub input: Vec<f64>,
    // pub output: Vec<f64>,
    fn forward(&mut self,input_data: Vec<f64>);
    fn backward(&mut self, error: Vec<f64>);
}

fn main() {
    let layer_sizes = [3, 5,3,4,5, 2]; // input, (4) hidden, output
    let nn = NeuralNetwork::new(layer_sizes.len(), &layer_sizes);

    let input = vec![0.5, -1.5, 1.0];
    let output = nn.feedforward(input);

    println!("Neural Network Output: {:?}", output);
}
