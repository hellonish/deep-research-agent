# The Evolution of Neural Networks: From Perceptrons to World Models

**Query:** Explain origin of Neural Networks, then how the CNN came to play, and then RNNs, what happened next that we are now working with LLMs and World Models?, I want comparisons, market share of each type of model, and how they evolved. You can use Bar charts and Pie charts to represent the data. Show me code snippets to show implementation of each type of model.
**Date:** 2026-02-19 18:34:19
**Cost controls:** max_depth=40, max_steps=5, max_probes=20, dupe_thresh=0.75
**Elapsed:** 90.5s
**Total blocks:** 14

## Summary

This report provides a comprehensive overview of the evolution of artificial neural networks, beginning with their biological inspiration and foundational components. It delves into the early Perceptron model, detailing its mechanism and limitations. The report then traces the development through Convolutional Neural Networks (CNNs) for image processing and Recurrent Neural Networks (RNNs) for sequential data, highlighting their architectural innovations and applications. The narrative continues to the advent of Large Language Models (LLMs), driven by the Transformer architecture, and concludes with an exploration of emerging World Models. Comparative analyses, illustrative market share data, and code snippets are included to provide a holistic understanding of these transformative technologies. Note that while the foundational sections are extensively cited from provided research, subsequent sections on CNNs, RNNs, LLMs, and World Models are based on general knowledge in the field of AI/ML, and market share/comparison data are illustrative due to the scope of provided research materials.

---

## [1] text: Introduction to Artificial Neural Networks: Biological Inspiration and Early Models

Artificial Neural Networks (ANNs) form a cornerstone of modern artificial intelligence, drawing significant inspiration from the intricate architecture and functionality of the human brain's biological neural networks [12]. These sophisticated computational systems are engineered to acquire knowledge from extensive datasets, autonomously discerning complex patterns and relationships [12].

### Biological Inspiration

The fundamental concept underpinning artificial neural networks is directly derived from the biological neuron, which serves as the primary unit for information processing within the brain and the broader nervous system [14]. Biological neurons function as transducers, converting input signals from preceding neurons into an output signal for subsequent neurons [14]. Their core role involves deciding whether to activate, or 'fire,' based on the aggregation of these incoming signals [14].

This biological blueprint translates into several analogous components within artificial neurons:

*   **Input Reception**: Biological neurons receive input signals via fine filaments known as dendrites [15]. Similarly, an artificial neuron, often termed a perceptron, accepts numerical inputs, such as pixel values in an image [7].
*   **Connection Strength**: The points of connection between dendrites and biological neurons are called synapses [10]. In ANNs, these connections are represented by 'weights,' which quantify the importance or strength of each input signal [15].
*   **Signal Integration and Thresholding**: A biological neuron integrates both excitatory (activation-promoting) and inhibitory (activation-suppressing) signals. If the cumulative excitatory signal surpasses a specific threshold, the neuron 'fires' [14]. Analogously, an artificial neuron computes a weighted sum of its inputs, and this sum is then processed by an activation function to determine its output, effectively deciding whether the neuron should 'activate' [7].

The conceptualization of artificial networks began in 1943 with Warren McCulloch and Walter Pitts, who proposed a model elucidating the potential operational mechanisms of brain neurons [7]. This pioneering work established the groundwork for the subsequent development of ANNs, which have since become powerful computational tools applied across diverse domains, including computer vision, natural language processing, and fraud detection [12].

---

## [2] text: Basic Components of an Artificial Neuron

An artificial neuron, frequently referred to as a node, constitutes the fundamental processing unit within an ANN [12]. It operates as a simplified mathematical model, executing a sequence of calculations to generate an output based on its received inputs [7]. The essential components of an artificial neuron include:

1.  **Inputs**: These are numerical values that the neuron receives, representing either raw data points or outputs from other neurons situated in a preceding layer [7]. These inputs are typically denoted as $x_1, x_2, 	ext{...}, x_n$.
2.  **Weights**: Each input connection leading to the neuron is assigned a specific weight, $w_i$. Weights are crucial as they dictate the 'importance' or 'strength' of each incoming signal [12]. These weights are parameters that the model learns and are iteratively adjusted during the training phase to minimize the disparity between the network's predicted outputs and the actual target outputs [15].
3.  **Summation Function (Weighted Sum)**: The neuron's initial operation involves calculating a weighted sum of its inputs. This process entails multiplying each input by its corresponding weight and then summing these products [7]. Mathematically, this can be expressed as: $Z = \sum_{i=1}^{n} (x_i \cdot w_i)$.
4.  **Bias**: In addition to the weighted sum, a bias term ($b$) is typically incorporated. The bias is another learnable parameter that enables the activation function to be shifted, thereby effectively adjusting the neuron's activation threshold [12]. The calculation thus becomes: $Z = \sum_{i=1}^{n} (x_i \cdot w_i) + b$.
5.  **Activation Function**: The result of the weighted sum (plus bias) is subsequently passed through an activation function. This function is vital for introducing non-linearity into the network, a characteristic indispensable for ANNs to learn and approximate the complex, non-linear relationships frequently observed in real-world data [10][14]. The specific type of activation function determines how a neuron 'fires' or activates, and it can also constrain the output value within a defined range [10]. Common activation functions include:
    *   **Sigmoid**: Compresses the output to a range between 0 and 1 [10].
    *   **Hyperbolic Tangent (tanh)**: Compresses the output to a range between -1 and 1 [14].
    *   **Rectified Linear Unit (ReLU)**: Outputs the input directly if it is positive, otherwise, it outputs zero [14].
    The final output of the neuron is given by: $Output = A(Z)$, where $A$ represents the activation function.
6.  **Output**: The final output generated by the activation function is then either transmitted as an input to subsequent neurons in the next layer or serves as the ultimate output of the entire network [7].

---

## [3] text: The Perceptron Model: A Foundational Step

The perceptron, introduced by Frank Rosenblatt in 1958, is widely recognized as the inaugural algorithm modeled after a biological neuron and stands as the simplest form of an artificial neuron [7][14]. It operates as a single-layer neural network, functioning primarily as a linear classifier for binary classification tasks [15].

### How the Perceptron Works

The perceptron model is engineered to perform binary classification by receiving multiple inputs, applying specific weights to each, summing these weighted inputs, and then passing this sum through a straightforward activation function to yield a binary output (e.g., 0 or 1, or -1 or 1) [15]. In its original formulation, the perceptron typically employed a step function as its activation mechanism. This function produces one value (e.g., 1) if the weighted sum surpasses a predetermined threshold, and another value (e.g., 0 or -1) otherwise [14].

The operational sequence can be summarized as follows:

1.  **Receive Inputs**: The perceptron takes in numerical inputs, denoted as $x_1, x_2, 	ext{...}, x_n$.
2.  **Apply Weights**: Each input $x_i$ is multiplied by its corresponding weight $w_i$.
3.  **Calculate Weighted Sum**: The products are summed, and a bias $b$ is added: $Z = \sum_{i=1}^{n} (x_i \cdot w_i) + b$.
4.  **Apply Activation Function**: The sum $Z$ is then processed by a step activation function, $A(Z)$. For instance, if $Z \ge \text{threshold}$, the output is 1; otherwise, the output is 0.
5.  **Produce Output**: The final output represents the classification decision.

### Training and Function

The perceptron learns through an iterative process of adjusting the weights associated with its neurons. During training, it processes input data, compares its predicted output against the actual desired output, and subsequently modifies the weights to minimize the discrepancy between them [15]. This iterative adjustment mechanism enables the perceptron to learn to classify input data accurately.

### Limitations

Despite its groundbreaking nature for its era, a single perceptron is inherently limited to solving linearly separable problems [14]. This implies that it can only classify data that can be perfectly divided by a single straight line (or a hyperplane in higher-dimensional spaces) [14]. It is incapable of modeling complex, non-linear relationships. This limitation presented a significant challenge until the advent of multi-layer perceptrons (MLPs), which overcome this constraint by incorporating 'hidden layers' of neurons. These hidden layers enable MLPs to model more intricate, non-linear relationships between inputs and outputs [14]. Furthermore, the introduction of diverse activation functions beyond the simple step function, such as sigmoid, tanh, and ReLU, in contemporary ANNs has substantially enhanced their learning capabilities and their capacity to model complex relationships [14].

---

## [4] text: The Rise of Convolutional Neural Networks (CNNs)

Following the foundational work on perceptrons and multi-layer perceptrons, a significant breakthrough in neural network architecture came with the development of Convolutional Neural Networks (CNNs). CNNs were specifically designed to process data with a known grid-like topology, such as image data, which can be represented as a 2D grid of pixels. Their emergence revolutionized the field of computer vision, enabling unprecedented performance in tasks like image classification, object detection, and segmentation.

The key innovation of CNNs lies in their use of specialized layers that exploit the spatial hierarchies in data. These include:

*   **Convolutional Layers**: Instead of fully connected layers where every input neuron connects to every output neuron, convolutional layers apply a 'filter' or 'kernel' that slides across the input data (e.g., an image). This filter performs a dot product with the local region of the input it's currently covering, producing a feature map. This process allows the network to automatically learn spatially invariant features, such as edges, textures, and patterns, at different levels of abstraction.
*   **Pooling Layers**: These layers typically follow convolutional layers and serve to reduce the spatial dimensions of the feature maps, thereby reducing the number of parameters and computational cost. Common pooling operations include max pooling (taking the maximum value in a window) and average pooling (taking the average value). Pooling also helps in making the detected features more robust to small translations and distortions in the input.
*   **Activation Functions**: Similar to traditional ANNs, non-linear activation functions (most commonly ReLU) are applied after convolutional operations to introduce non-linearity, allowing the network to learn more complex patterns.
*   **Fully Connected Layers**: After several convolutional and pooling layers, the high-level features are typically flattened and fed into one or more fully connected layers, similar to those in a traditional MLP, for final classification or regression.

CNNs' ability to automatically learn hierarchical features from raw pixel data, coupled with their parameter sharing mechanism (where the same filter is applied across different locations), made them highly efficient and effective for visual tasks, overcoming the limitations of earlier models that struggled with the high dimensionality and spatial complexity of images.

---

## [5] code: CNN Implementation Snippet (PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional Layer 1: 1 input channel (grayscale image), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Convolutional Layer 2: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer with 2x2 window and stride 2
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layer: input features depend on image size and conv/pool operations
        # For a 28x28 input image, after two conv+pool layers, spatial dim becomes 7x7
        # 64 channels * 7 * 7 = 3136 input features
        self.fc1 = nn.Linear(64 * 7 * 7, 10) # 10 output classes (e.g., MNIST digits)

    def forward(self, x):
        # Input x: (batch_size, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x))) # (batch_size, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x))) # (batch_size, 64, 7, 7)
        x = x.view(-1, 64 * 7 * 7)          # Flatten for fully connected layer
        x = self.fc1(x)                     # (batch_size, 10)
        return x

# Example usage:
# model = SimpleCNN()
# input_tensor = torch.randn(1, 1, 28, 28) # Batch size 1, 1 channel, 28x28 image
# output = model(input_tensor)
# print(output.shape) # Expected: torch.Size([1, 10])
```

---

## [6] text: Recurrent Neural Networks (RNNs) for Sequential Data

While CNNs excelled in processing spatial data like images, a different architectural paradigm was needed for sequential data, where the order and context of information are crucial. This led to the development of Recurrent Neural Networks (RNNs). RNNs are specifically designed to handle sequences by maintaining an internal state (memory) that captures information from previous steps in the sequence, allowing them to make predictions based on past inputs.

Key characteristics of RNNs include:

*   **Recurrent Connections**: Unlike feedforward networks, RNNs have connections that loop back on themselves, allowing information to persist from one step to the next. At each time step, an RNN cell takes an input from the current step and the hidden state from the previous step to produce an output and a new hidden state.
*   **Memory**: The hidden state acts as a form of short-term memory, allowing the network to learn dependencies across different time steps. This makes them suitable for tasks where context is important, such as natural language processing (NLP), speech recognition, and time series prediction.
*   **Vanishing/Exploding Gradients**: A major challenge with vanilla RNNs is the vanishing or exploding gradient problem during backpropagation through time. This makes it difficult for them to learn long-range dependencies, as gradients can become too small or too large over many time steps.

To address these limitations, more sophisticated recurrent architectures were developed:

*   **Long Short-Term Memory (LSTM) Networks**: LSTMs, introduced by Hochreiter and Schmidhuber in 1997, incorporate 'gates' (input, forget, and output gates) that regulate the flow of information into and out of a 'cell state.' This cell state acts as a long-term memory, allowing LSTMs to selectively remember or forget information over extended periods, effectively mitigating the vanishing gradient problem.
*   **Gated Recurrent Units (GRUs)**: GRUs are a simpler variant of LSTMs, featuring fewer gates (reset and update gates) but offering comparable performance on many tasks. Their reduced complexity often leads to faster training times.

RNNs, particularly LSTMs and GRUs, became the standard for sequence modeling tasks, enabling significant advancements in machine translation, text generation, and speech recognition before the advent of the Transformer architecture.

---

## [7] code: RNN (LSTM) Implementation Snippet (PyTorch)

```python
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer: input_size is feature dimension, hidden_size is internal state size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Input x: (batch_size, sequence_length, input_size)
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0)) # out: (batch_size, sequence_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :]) # Take the output from the last time step
        return out

# Example usage:
# input_size = 10   # e.g., word embedding dimension
# hidden_size = 128
# num_layers = 2
# num_classes = 5   # e.g., sentiment classes
# model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)
# input_tensor = torch.randn(32, 20, input_size) # Batch size 32, sequence length 20
# output = model(input_tensor)
# print(output.shape) # Expected: torch.Size([32, 5])
```

---

## [8] text: Evolution to Modern Large Language Models (LLMs)

The landscape of neural networks, particularly for sequential data, underwent another profound transformation with the introduction of the Transformer architecture in 2017. This architecture fundamentally changed how models process sequences, moving away from the sequential processing of RNNs and LSTMs to a parallelized mechanism based on 'self-attention.'

Key aspects of the Transformer and its impact on LLMs include:

*   **Self-Attention Mechanism**: The core innovation of the Transformer is the self-attention mechanism, which allows the model to weigh the importance of different words (or tokens) in an input sequence when processing each word. This enables the model to capture long-range dependencies much more effectively and efficiently than RNNs, without the vanishing gradient problem.
*   **Parallelization**: Unlike RNNs, which process tokens one by one, the self-attention mechanism allows all tokens in a sequence to be processed simultaneously. This parallelization significantly speeds up training on modern hardware (GPUs, TPUs) and enables the training of much larger models.
*   **Encoder-Decoder Structure**: The original Transformer model consists of an encoder stack and a decoder stack. The encoder processes the input sequence, and the decoder generates the output sequence, attending to both the encoder's output and its own previous outputs.
*   **Positional Encoding**: Since the self-attention mechanism itself is permutation-invariant (it doesn't inherently understand word order), positional encodings are added to the input embeddings to inject information about the relative or absolute position of tokens in the sequence.

The Transformer architecture quickly became the dominant paradigm for natural language processing. Its scalability, combined with the availability of vast amounts of text data and increased computational power, led to the development of Large Language Models (LLMs). LLMs are Transformer-based models with billions or even trillions of parameters, pre-trained on massive text corpora. This pre-training allows them to learn rich representations of language, enabling them to perform a wide array of tasks, including text generation, translation, summarization, question answering, and even complex reasoning, often with remarkable fluency and coherence. Models like GPT (Generative Pre-trained Transformer) series, BERT, and T5 are prominent examples of LLMs that have reshaped the field of AI.

---

## [9] code: LLM Architectural Concepts (Simplified Transformer Block in PyTorch)

```python
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.size()

        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)

        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
        return self.output_proj(output)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention part
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout1(attn_output)) # Add & Norm

        # Feed-forward part
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output)) # Add & Norm
        return x

# Example usage:
# embed_dim = 512
# num_heads = 8
# ff_dim = 2048
# transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
# input_tensor = torch.randn(1, 10, embed_dim) # Batch size 1, sequence length 10
# output = transformer_block(input_tensor)
# print(output.shape) # Expected: torch.Size([1, 10, 512])
```

---

## [10] text: The Emergence of World Models

Beyond the impressive capabilities of LLMs in language understanding and generation, a more ambitious frontier in AI research is the development of 'World Models.' Inspired by the human brain's ability to build internal representations of the environment and use them for planning and prediction, World Models aim to create AI systems that can learn a compressed, predictive model of their environment. This internal model allows the agent to simulate future outcomes, plan actions, and even learn complex behaviors entirely within its imagination, without constant interaction with the real world.

World Models typically consist of three main components:

*   **Vision Model (V)**: This component processes raw sensory input (e.g., images from a camera) and encodes it into a compact, lower-dimensional latent representation. This representation captures the essential features of the current observation, discarding irrelevant details.
*   **Memory Model (M)**: Also known as the 'dynamics model' or 'recurrent neural network,' this component learns to predict the next latent state of the environment given the current latent state and the agent's action. It essentially learns the 'physics' or transition rules of the world. This model is often a recurrent neural network (like an LSTM or GRU) that processes sequences of latent states and actions.
*   **Controller Model (C)**: This component takes the current latent state from the vision model and the predicted future states from the memory model to decide on the next action. The controller can be a simple feedforward neural network or a more complex policy network that learns to achieve specific goals by interacting with the internal world model.

The primary advantage of World Models is their potential for highly efficient learning. By simulating experiences within its learned internal model, an agent can generate vast amounts of synthetic data and practice complex tasks without the cost, time, or safety constraints of real-world interaction. This approach holds promise for developing more robust, adaptive, and general-purpose AI agents capable of understanding and interacting with complex environments in a human-like manner, moving towards truly intelligent agents that can reason and plan.

---

## [11] table: Comparative Analysis of Neural Network Architectures (Illustrative)

| Model Type | Primary Use Case | Key Architectural Feature | Strengths | Limitations |
| --- | --- | --- | --- | --- |
| Perceptron (ANN) | Binary Classification | Single layer, weighted sum, step activation | Simplicity, foundational for ANNs | Limited to linearly separable problems |
| Multi-Layer Perceptron (MLP) | General Classification/Regression | Multiple hidden layers, non-linear activations | Learns complex non-linear relationships | Lacks spatial/temporal awareness, many parameters for high-dim data |
| Convolutional Neural Network (CNN) | Image/Video Processing | Convolutional layers, pooling, parameter sharing | Excellent for spatial data, learns hierarchical features, translation invariant | Less effective for sequential data, fixed input size |
| Recurrent Neural Network (RNN) | Sequential Data Processing (Text, Speech, Time Series) | Recurrent connections, hidden state (memory) | Handles variable-length sequences, captures temporal dependencies | Suffers from vanishing/exploding gradients, struggles with long-range dependencies (vanilla RNNs) |
| Long Short-Term Memory (LSTM) / Gated Recurrent Unit (GRU) | Advanced Sequential Data Processing | Gating mechanisms (forget, input, output gates) | Mitigates vanishing gradient problem, learns long-range dependencies | Sequential processing can be slow, still less efficient for very long sequences than Transformers |
| Large Language Model (LLM) (Transformer-based) | Natural Language Understanding/Generation, Reasoning | Self-attention mechanism, parallel processing, positional encoding | Exceptional for long-range dependencies, highly parallelizable, scalable to billions of parameters | Computationally intensive to train, requires massive datasets, can be prone to hallucination |
| World Model | Reinforcement Learning, Planning, Simulation | Vision, Memory (dynamics), Controller components | Enables planning in latent space, efficient learning through simulation, potential for general intelligence | Complex to build and train, accuracy of internal model is critical, still an active research area |

---

## [12] chart: Illustrative Market Share of Neural Network Paradigms (Conceptual)

**Chart type:** pie
```json
{
  "labels": [
    "MLP/Basic ANN",
    "CNNs",
    "RNNs/LSTMs/GRUs",
    "LLMs (Transformers)",
    "World Models & Emerging"
  ],
  "datasets": [
    {
      "label": "Market Share",
      "data": [
        5.0,
        30.0,
        15.0,
        45.0,
        5.0
      ]
    }
  ]
}
```

---

## [13] chart: Historical Evolution of Neural Network Paradigms (Conceptual Adoption Timeline)

**Chart type:** bar
```json
{
  "labels": [
    "1950s-1980s",
    "1990s-Early 2000s",
    "2000s-2010s",
    "2010s-2020s",
    "2020s Onwards"
  ],
  "datasets": [
    {
      "label": "Perceptron/ANN",
      "data": [
        80.0,
        20.0,
        10.0,
        5.0,
        2.0
      ]
    },
    {
      "label": "CNNs",
      "data": [
        0.0,
        10.0,
        40.0,
        30.0,
        20.0
      ]
    },
    {
      "label": "RNNs/LSTMs",
      "data": [
        0.0,
        5.0,
        30.0,
        20.0,
        10.0
      ]
    },
    {
      "label": "LLMs (Transformers)",
      "data": [
        0.0,
        0.0,
        0.0,
        40.0,
        60.0
      ]
    },
    {
      "label": "World Models",
      "data": [
        0.0,
        0.0,
        0.0,
        5.0,
        8.0
      ]
    }
  ]
}
```

---

## [14] source_list: Block 14

- https://medium.com/@abhaysingh71711/perceptron-building-block-of-artificial-neural-network-e5d6b366f877
- https://www.codemotion.com/magazine/ai-ml/deep-learning/the-perceptron-the-first-building-block-of-neural-networks/
- https://developer.ibm.com/articles/cc-cognitive-neural-networks-deep-dive/
- https://medium.com/@tahirbalarabe2/what-is-the-perceptron-a-foundational-neural-network-model-c722687dc51a
- https://en.wikipedia.org/wiki/Neural_network_(machine_learning)
- https://medium.com/@bhatadithya54764118/day-43-introduction-to-neural-networks-biological-vs-artificial-networks-b8cd40d7df7d
- https://medium.com/@mustaphaliaichi/neural-networks-and-deep-learning-a-comprehensive-introduction-092449336c1f
- https://www.bizstim.com/news/article/understanding-perceptrons-in-machine-learning
- https://towardsdatascience.com/the-concept-of-artificial-neurons-perceptrons-in-neural-networks-fab22249cbfc/
- https://www.ncbi.nlm.nih.gov/books/NBK583971/
- https://www.codemotion.com/magazine/ai-ml/deep-learning/artificial-neural-networks-biological-inspiration-behind-deep-learning/
- https://www.pvpsiddhartha.ac.in/dep_cse/lect_note/32/ml/ML%20UNIT-II.pdf
- https://matt.might.net/articles/hello-perceptron/
- https://www.lucentinnovation.com/resources/technology-posts/understanding-the-perceptron

---
