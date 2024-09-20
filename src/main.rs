#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
use rand::Rng;
use std::f32::{self,consts::E};

struct Math;

impl Math {
    fn exp(x: f32) -> f32 { E.powf(x) } // without `use std::f32::self` you'd use `f32::pow(E,x)`

    fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + Self::exp(-x)) }

    fn sigmoid_prime(x: f32) -> f32 {
        let sig = Self::sigmoid(x);
        sig * (1.0 - sig)
    }

    fn ReLU(x: f32) -> f32 {
        if x > 0.0 { x } else { 0.0 }
    }

    fn ReLUPrime(x: f32) -> f32 {
        if x >= 0.0 { 1.0 } else { 0.0 }
    }

    fn leaky_ReLU(x: f32) -> f32 {
        if x > 0.0 { x }  else { 0.01 * x }
    }

    fn leaky_ReLU_prime(x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.01 }
    }

    fn tanh(x: f32) -> f32 {
        (Self::exp(x) - Self::exp(-x)) / (Self::exp(x) + Self::exp(-x))
    }

    fn tanh_prime(x: f32) -> f32 {
        1.0 - f32::powf(Self::tanh(x), 2.0)
    }

    fn mult_scalar_and_vector(v: &[f32], s: f32) -> Vec<f32> {
        v.iter().map(|x| s*x).collect()
    }

    fn transpose(v: &[Vec<f32>]) -> Vec<Vec<f32>> {
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

    fn mean_squared_error(predictions: &[f32], targets: &[f32]) -> f32 {
	predictions.iter().zip(targets)
	    .map(|(p, t)| (p - t).powi(2))
	    .sum::<f32>() / predictions.len() as f32
    }

    fn binary_cross_entropy(predictions: &[f32], targets: &[f32]) -> f32 {
        predictions.iter().zip(targets)
            .map(|(p, t)| { if *t == 1.0 { -p.ln() } else { -(1.0 - p).ln() } } )
            .sum::<f32>() / predictions.len() as f32
    }
    
    fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
        v1.iter().zip(v2.iter())
            .map(|(x, y)| x * y)
            .sum()
    }
}

struct NeuralNetwork {
    // pub biases: Vec<f32>,
    // pub weights: Vec<Vec<f32>>,
    layers: Vec<Box<dyn Layer>> // we use `dyn` here because Layer is a Trait Object
}


trait Layer {
    // pub input: Vec<f32>,
    // pub output: Vec<f32>,
    fn forward(&mut self,input_data: Vec<f32>);
    fn backward(&mut self, error: Vec<f32>);
    fn output(&self) -> Vec<f32>;
    fn get_weights(&self) -> Vec<Vec<f32>>;
    fn get_biases(&self) -> Vec<f32>;
}

struct InputLayer {
    input: Vec<f32>
}

impl Layer for InputLayer {
    fn forward(&mut self,input_data: Vec<f32>) {
	self.input = input_data;
    }

    fn backward(&mut self, error: Vec<f32>) {
        // The input layer does not need/use backprop...
    }

    fn output(&self) -> Vec<f32> {
        self.input.clone() // No changes reqd in the input layer
    }
    fn get_weights(&self) -> Vec<Vec<f32>> {
	vec![] // input layer does not have any weights
    }
    fn get_biases(&self) -> Vec<f32> {
	vec![] // no biases in the input layer 
    }
}

struct HiddenLayer {
    input: Vec<f32>,
    output: Vec<f32>,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
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
    fn forward(&mut self, input_data: Vec<f32>) {
        self.input = input_data.clone();
        self.output.clear();

        for (weight, bias) in self.weights.iter().zip(self.biases.iter()) {
            let z = Math::dot_product(&self.input, weight) + bias;
            self.output.push(Math::sigmoid(z));
        }
    }

    fn backward(&mut self, error: Vec<f32>) {
        // Calculate the gradient with respect to weights and biases
        let gradients: Vec<f32> = self.output.iter()
            .zip(error.iter())
            .map(|(o, e)| Math::sigmoid_prime(*o) * e)
            .collect();

        // Update weights and biases based on gradients (using a learning rate)
        let learning_rate = 0.01;
        for (i, (grad, bias)) in gradients.iter().zip(self.biases.iter_mut()).enumerate() {
            *bias -= learning_rate * grad;
            for (j, weight) in self.weights[i].iter_mut().enumerate() {
                *weight -= learning_rate * grad * self.input[j];
            }
        }
    }

    fn output(&self) -> Vec<f32> {
        self.output.clone()
    }

    fn get_weights(&self) -> Vec<Vec<f32>> {
        self.weights.clone()
    }

    fn get_biases(&self) -> Vec<f32> {
        self.biases.clone()
    }
}

struct OutputLayer {
    input: Vec<f32>,
    output: Vec<f32>,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
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
    fn forward(&mut self, input_data: Vec<f32>) {
        self.input = input_data.clone();
        self.output.clear();

        for (weight, bias) in self.weights.iter().zip(self.biases.iter()) {
            let z = Math::dot_product(&self.input, weight) + bias;
            self.output.push(Math::sigmoid(z)); // Using sigmoid activation
        }
    }

    fn backward(&mut self, error: Vec<f32>) {
        let gradients: Vec<f32> = self.output.iter()
            .zip(error.iter())
            .map(|(o, e)| Math::sigmoid_prime(*o) * e)
            .collect();

        let learning_rate = 0.01;
        for (i, (grad, bias)) in gradients.iter().zip(self.biases.iter_mut()).enumerate() {
            *bias -= learning_rate * grad;
            for (j, weight) in self.weights[i].iter_mut().enumerate() {
                *weight -= learning_rate * grad * self.input[j];
            }
        }
    }

    fn output(&self) -> Vec<f32> {
        self.output.clone()
    }

    fn get_weights(&self) -> Vec<Vec<f32>> {
        self.weights.clone()
    }

    fn get_biases(&self) -> Vec<f32> {
        self.biases.clone()
    }
}

impl NeuralNetwork {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut nn = NeuralNetwork {
            layers: Vec::new(),
        };

        for i in 0..layer_sizes.len() {
            if i == 0 {
                nn.layers.push(Box::new(InputLayer { input: vec![] }));
            } else if i == layer_sizes.len() - 1 {
                nn.layers.push(Box::new(OutputLayer::new(layer_sizes[i], layer_sizes[i - 1])));
            } else {
                nn.layers.push(Box::new(HiddenLayer::new(layer_sizes[i], layer_sizes[i - 1])));
            }
        }

        nn
    }

    pub fn feedforward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let mut current_input = input;
        for layer in &mut self.layers {
            layer.forward(current_input.clone());
            current_input = layer.output(); // Update input for the next layer
        }
        current_input // Return output from the last layer
    }

    pub fn predict(&mut self, input: Vec<f32>) -> Vec<f32> {
        self.feedforward(input)
    }

    pub fn backprop(&mut self, target: Vec<f32>) {
        let output = self.layers.last_mut().unwrap().output();
        let error: Vec<f32> = output.iter()
            .zip(target.iter())
            .map(|(o, t)| o - t)
            .collect();

        // Backpropagate the error through the layers
        for layer in self.layers.iter_mut().rev() {
            layer.backward(error.clone());
        }
    }

    pub fn fit(&mut self, inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>, epochs: usize) {
        for _ in 0..epochs {
            for (input, target) in inputs.iter().zip(&targets) {
                let _ = self.feedforward(input.clone());
                self.backprop(target.clone());
            }
        }
    }
}

fn main() {
    let layer_sizes = [3, 4, 2]; // Input, hidden, output
    let mut nn = NeuralNetwork::new(&layer_sizes);

    // Example input and target for training
    let inputs = vec![
        vec![0.5, -1.5, 1.0],
        vec![1.0, 0.0, -1.0],
    ];
    let targets = vec![
        vec![0.0, 1.0],
        vec![1.0, 0.0],
    ];

    // Training the neural network
    nn.fit(inputs.clone(), targets.clone(), 1_000_000); // if we don't use clone here the model starts from altered data each time, so it will start to learn incorrectly from subsequent epochs

    // Predicting
    let input = vec![0.5, -1.5, 1.0];
    let output = nn.predict(input);

    println!("Neural Network Output: {:?}", output);
}

// fn main() {
//     let layer_sizes = [3, 5,3,4,5, 2]; // input, (4) hidden, output
//     let mut nn = NeuralNetwork::new(&layer_sizes);

//     let input = vec![0.5, -1.5, 1.0];
//     let output = nn.feedforward(input);

//     println!("Neural Network Output: {:?}", output);
// }
