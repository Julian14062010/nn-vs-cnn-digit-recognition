# NN vs CNN Digit Recognition

This project compares different neural network approaches for handwritten digit recognition and demonstrates their performance on both standard datasets and custom user input.

## Overview

The goal of this project is to explore the differences between:

- Neural Network (from scratch, pure Python)
- Convolutional Neural Network (from scratch, pure Python)
- Neural Network using PyTorch
- CNN using PyTorch

All models are trained on the MNIST dataset and evaluated on both test data and custom drawn digits.

---

## Interactive Demo

The project includes an interactive drawing application.

You can:
- Draw a digit (0–9)
- Let the trained model predict it in real time

Run:

```bash
python main.py

Key Insight

The CNN performs significantly better than a standard neural network on hand-drawn digits.

Reason:
CNNs capture spatial structure and are more robust to small shifts in x and y direction.
This makes them better suited for real-world handwritten input compared to fully connected networks.

Results
Neural Network: ~97% accuracy
CNN: up to 99.4% accuracy

Detailed analysis shows:

Some digits (e.g. 9, 2) are more difficult to classify
Misclassifications often occur between visually similar digits (e.g. 3 ↔ 5, 4 ↔ 9)

This highlights the importance of spatial feature extraction.

From Scratch vs Framework
From Scratch (pure Python)

Advantages:

Full understanding of how neural networks work
Complete control over implementation

Disadvantages:

Slower
More complex to implement
Harder to scale
PyTorch

Advantages:

Fast and efficient
Easy to experiment with architectures
Built-in tools for training and evaluation

Disadvantages:

Less low-level control
Abstraction can hide implementation details

Dataset

MNIST is automatically downloaded when running the code (via PyTorch).

Setup

Install dependencies:

pip install torch torchvision arcade
Project Structure
models/     # neural network implementations
app/        # drawing interface
main.py     # entry point
Conclusion

This project demonstrates how architectural choices impact performance in machine learning.

While simple neural networks already achieve strong results, CNNs provide a clear advantage when dealing with spatial data such as images.