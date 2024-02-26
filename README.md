
---

## Introduction

This library contains a set of functions for encoding images into a quantum state using the Flexible Representation of Quantum Images (FRQI) and Novel Enhanced Quantum Representation (NEQR) methods, as well as for decoding states into images. Import this module into your project or Jupyter Notebook using the command `import frqi`. Below is a brief description of the functions that may be useful to you.

## Function Descriptions

- `image_preparation(image, num_qubits=None)`: Prepares your image for encoding, making it square and grayscale. If you specify the number of qubits, it will resize the image accordingly.
  - 5 qubits - image size 4x4
  - 7 qubits - image size 8x8
  - 9 qubits - image size 16x16
  - 11 qubits - image size 32x32

- `make_circ(image, qubit)`: Returns a Qiskit circuit for encoding the image using the FRQI method.

- `get_state(qc)`: Returns the statevector.

- `get_counts(qc, qubit, shots=None)`: Performs measurement and returns Qiskit counts.

- `decoding(counts, num_qubits=None)`: Takes either a statevector or counts and returns the image as an array, which can be visualized using `plt.imshow(decoding(state), cmap='gray')`.

- `calculate_mse(image1, image2)` and `calculate_ssim(image1, image2)`: Metrics for comparing similarity.

- `create_statevector(image, qubits)`: Returns the state vector for the image using the FRQI method without constructing a Qiskit circuit.

- `encode_image_neqr(image)`: Encodes the image into a state vector using the NEQR method.

- `decode_neqr(statevector)`: Decodes the state vector into an image.

---

Feel free to include this README in your repository for clear documentation of your library's functionalities. If you have any further questions or need assistance, feel free to ask!
