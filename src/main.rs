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
    fn exp(x: f64) -> f64 { f64::powf(E, x) }

    fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + Self::exp(-x)) }

    fn sigmoid_prime(x: f64) -> f64 {
        let sig = Self::sigmoid(x);
        sig * (1.0 - sig)
    }

    fn ReLU(x: f64) -> f64 {
        if x > 0.0 { x } else { 0.0 }
    }

    fn ReLUPrime(x: f64) -> f64 {
        if x >= 0.0 { 1.0 } else { 0.0 }
    }

    fn leaky_ReLU(x: f64) -> f64 {
        if x > 0.0 { x }  else { 0.01 * x }
    }

    fn leaky_ReLU_prime(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.01 }
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
        v.iter().map(|x| s*x).collect()
    }

    fn transpose(v: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if v.is_empty() || v[0].is_empty() { return vec![]; }

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
    // pub biases: Vec<f64>,
    // pub weights: Vec<Vec<f64>>,
    layers: Vec<Box<dyn Layer>> // we use `dyn` here because Layer is a Trait Object
}

impl NeuralNetwork {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut nn = NeuralNetwork {
            layers: Vec::new(),
        };

        for i in 0..layer_sizes.len() {
            if i == 0 {
                // Input layer
                nn.layers.push(Box::new(InputLayer { input: vec![] }));
            } else if i == layer_sizes.len() - 1 {
                // Output layer
                nn.layers.push(Box::new(OutputLayer::new(layer_sizes[i], layer_sizes[i - 1])));
            } else {
                // Hidden layer
                nn.layers.push(Box::new(HiddenLayer::new(layer_sizes[i], layer_sizes[i - 1])));
            }
        }

        nn
    }

    pub fn feedforward(&mut self, input: Vec<f64>) -> Vec<f64> {
        let mut current_input = input;
        for layer in &mut self.layers {
            layer.forward(current_input.clone());
            current_input = layer.output(); // Update input for the next layer
        }
        current_input // Return output from the last layer
    }
}


trait Layer {
    // pub input: Vec<f64>,
    // pub output: Vec<f64>,
    fn forward(&mut self,input_data: Vec<f64>);
    fn backward(&mut self, error: Vec<f64>);
    fn output(&self) -> Vec<f64>;
}

struct InputLayer {
    input: Vec<f64>
}

impl Layer for InputLayer {
    fn forward(&mut self,input_data: Vec<f64>) {
	self.input = input_data;
    }

    fn backward(&mut self, error: Vec<f64>) {
        todo!() // The input layer does not need/use backprop...
    }

    fn output(&self) -> Vec<f64> {
        self.input.clone() // No changes reqd in the input layer
    }
}

struct HiddenLayer {
    input: Vec<f64>,
    output: Vec<f64>,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

impl HiddenLayer {
    pub fn new(size: usize, prev_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..size).map(|_| (0..prev_size).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect();
        let biases = (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        
        HiddenLayer {
            input: vec![],
            output: vec![],
            weights,
            biases,
        }
    }
}

impl Layer for HiddenLayer {
    fn forward(&mut self, input_data: Vec<f64>) {
        self.input = input_data.clone();
        self.output.clear();

        for (weight, bias) in self.weights.iter().zip(self.biases.iter()) {
            let z = Math::dot_product(&self.input, weight) + bias;
            self.output.push(Math::sigmoid(z)); // Using sigmoid activation
        }
    }

    fn backward(&mut self, error: Vec<f64>) {
        // Implement backward logic here if needed
    }

    fn output(&self) -> Vec<f64> {
        self.output.clone()
    }
}

struct OutputLayer {
    input: Vec<f64>,
    output: Vec<f64>,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

impl OutputLayer {
    pub fn new(size: usize, prev_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..size).map(|_| (0..prev_size).map(|_| rng.gen_range(-1.0..1.0)).collect()).collect();
        let biases = (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        
        OutputLayer {
            input: vec![],
            output: vec![],
            weights,
            biases,
        }
    }
}
impl Layer for OutputLayer {
    fn forward(&mut self, input_data: Vec<f64>) {
        self.input = input_data.clone();
        self.output.clear();

        for (weight, bias) in self.weights.iter().zip(self.biases.iter()) {
            let z = Math::dot_product(&self.input, weight) + bias;
            self.output.push(Math::sigmoid(z)); // Using sigmoid activation
        }
    }

    fn backward(&mut self, error: Vec<f64>) {
        // Implement backward logic here if needed
    }

    fn output(&self) -> Vec<f64> {
        self.output.clone()
    }
}

fn main() {
    let layer_sizes = [3, 5,3,4,5, 2]; // input, (4) hidden, output
    let mut nn = NeuralNetwork::new(&layer_sizes);

    let input = vec![0.5, -1.5, 1.0];
    let output = nn.feedforward(input);

    println!("Neural Network Output: {:?}", output);
}
