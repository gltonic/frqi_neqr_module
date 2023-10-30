from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, QuantumRegister
from qiskit.qasm import pi
from qiskit import execute, Aer, BasicAer
import numpy as np
import random
from qiskit.circuit.library.standard_gates import RYGate, RYYGate
from qiskit.quantum_info import Statevector, state_fidelity,entropy,shannon_entropy, partial_trace
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
import math
from skimage.metrics import mean_squared_error, structural_similarity
from PIL import Image


# The `hadamard` function applies the Hadamard gate operation to the specified qubits in the circuit.
# Args:
#   - `circ`: QuantumCircuit, the quantum circuit to which the Hadamard gate is applied.
#   - `n`: List of qubits on which the Hadamard gate is applied.
def hadamard(circ, n):
    for i in n:
        circ.h(i)

# The `change` function determines the difference between two quantum states.
# Args:
#   - `state`: str, the initial quantum state represented as a string.
#   - `new_state`: str, the new quantum state represented as a string.
# Returns:
#   - An array of indices of qubits where a change occurred between `state` and `new_state`.
def change(state, new_state):
    n = len(state)
    c = np.array([])

    for i in range(n):
        if state[i] != new_state[i]:
            c = np.append(c, int(i))

    if len(c) > 0:
        return c.astype(int)
    else:
        return c

# The `binary` function performs quantum state changes in the circuit based on the differences between `state` and `new_state`.
# Args:
#   - `circ`: QuantumCircuit, the quantum circuit.
#   - `state`: str, the initial quantum state represented as a string.
#   - `new_state`: str, the new quantum state represented as a string.
#   - `num_qubits`: int, the number of qubits in the circuit.
def binary(circ, state, new_state, num_qubits):
    c = change(state, new_state)
    if len(c) > 0:
        circ.x(np.abs(c - (num_qubits - 2)))
    else:
        pass

# The `cnri` function performs a controlled rotation around the Y-axis (CRY) operation on the specified qubits.
# Args:
#   - `circ`: QuantumCircuit, the quantum circuit to which the CRY operation is applied.
#   - `n`: List of control qubits.
#   - `t`: Target qubit.
#   - `theta`: Rotation angle.
def cnri(circ, n, t, theta):
    controls = len(n)
    cry = RYGate(2 * theta).control(controls)
    aux = np.append(n, t).tolist()
    circ.append(cry, aux)

# The `frqi` function performs a quantum transformation based on the given angles and qubit states.
# Args:
#   - `circ`: QuantumCircuit, the quantum circuit.
#   - `n`: List of qubits for Hadamard operations.
#   - `t`: Target qubit for CRY operation.
#   - `angles`: List of angles for CRY operations.
def frqi(circ, n, t, angles):
    hadamard(circ, n)
    num_qubits = circ.num_qubits
    binary_length = num_qubits - 1
    j = 0
    for i in angles:
        state = '{0:0{1}b}'.format(j - 1, binary_length)
        new_state = '{0:0{1}b}'.format(j, binary_length)
        if j == 0:
            cnri(circ, n, t, i)
        else:
            binary(circ, state, new_state, num_qubits)
            cnri(circ, n, t, i)
        j += 1

# The `decoding` function reconstructs an image from quantum counts.
# Args:
#   - `counts`: Quantum counts or state vector.
#   - `num_qubits`: The number of qubits in the quantum circuit (optional).
# Returns:
#   - A reconstructed image as a NumPy array.

def decoding(counts, num_qubits=None):
    # Initialize an empty NumPy array to store the reconstructed image
    retrieved_image = np.array([])
    
    # Check if the input counts are a NumPy array or Statevector
    if type(counts) == np.ndarray or type(counts) == Statevector:
        # If num_qubits is not specified, calculate it based on the length of counts
        num_qubits = int(math.log(len(counts), 2))
        
        # Calculate the number of pixels and the side of the picture
        pixels = 2 ** (num_qubits - 1)
        picture_side = int(np.sqrt(pixels))
        binary_length = num_qubits - 1
        
        # Convert counts to a dictionary if it's a Statevector
        counts = Statevector(counts).to_dict()
        
        # Iterate through all possible states and retrieve counts
        for i in range(pixels):
            try:
                s = format(i, '0{0}b'.format(binary_length))
                new_s = '1' + s
                retrieved_image = np.append(retrieved_image, counts[new_s])
            except KeyError:
                retrieved_image = np.append(retrieved_image, [0.0])
    else:
        # If num_qubits is specified, use it to calculate the number of pixels and the side of the picture
        pixels = 2 ** (num_qubits - 1)
        picture_side = int(np.sqrt(pixels))
        binary_length = num_qubits - 1
        
        # Get the number of shots from the counts
        shots = counts.shots()
        
        # Iterate through all possible states and retrieve counts with normalization
        for i in range(pixels):
            try:
                s = format(i, '0{0}b'.format(binary_length))
                new_s = '1' + s
                retrieved_image = np.append(retrieved_image, np.sqrt(counts[new_s] / shots))
            except KeyError:
                retrieved_image = np.append(retrieved_image, [0.0])
    
    # Convert the retrieved image to real numbers
    retrieved_image = np.real(retrieved_image)
    
    # Scale the retrieved image to match the image's original intensity range
    retrieved_image *= picture_side * 255.0
    
    # Convert the retrieved image to integer values
    retrieved_image = retrieved_image.astype('int')
    
    # Reshape the retrieved image to its original dimensions
    retrieved_image = retrieved_image.reshape((picture_side, picture_side))
    
    return retrieved_image

# The `get_angles` function calculates angles from an image for encoding.
# Args:
#   - `im`: Image data (NumPy array or PIL Image).
#   - `qubit`: Number of qubits.
# Returns:
#   - List of angles.

def get_angles(im, qubit):
    # Calculate the number of pixels based on the number of qubits
    pixels = 2 ** (qubit - 1)
    
    # Check the type of input image data
    if type(im) == np.ndarray:
        # If the input is a NumPy array, reshape it to a 1D array of pixel values
        pixel_values = im.reshape(pixels)
        # Normalize pixel values to the range [0, 1]
        normalized_pixels = pixel_values / 255.0
        # Calculate angles using arcsin
        angles = np.arcsin(normalized_pixels)
    elif type(im) == Image.Image:
        # If the input is a PIL Image, convert it to a NumPy array
        im1 = np.asarray(im)
        # Reshape the NumPy array to a 1D array of pixel values
        pixel_values = im1.reshape(pixels)
        # Normalize pixel values to the range [0, 1]
        normalized_pixels = pixel_values / 255.0
        # Calculate angles using arcsin
        angles = np.arcsin(normalized_pixels)
    
    return angles
    
# The `make_circ` function creates a quantum circuit for encoding an image.
# Args:
#   - `image`: Image data (PIL Image or NumPy array).
#   - `qubit`: Number of qubits.
# Returns:
#   - Quantum circuit.

def make_circ(image, qubit):
    # Create a QuantumRegister and a ClassicalRegister with 'qubit' qubits
    qr = QuantumRegister(qubit, 'q')
    cr = ClassicalRegister(qubit, 'c')
    
    # Create a QuantumCircuit with the specified registers
    qc = QuantumCircuit(qr, cr)
    
    # Calculate angles from the image for encoding
    angles = get_angles(image, qubit)
    
    # Set the target qubit 't' as the last qubit and create a list of control qubits 'n'
    t = qubit - 1
    n = [i for i in range(t)]
    
    # Apply the Quantum Fourier Resampling (FRQI) transformation to the circuit
    frqi(qc, n, t, angles)
    
    return qc

# The `get_state` function obtains the state vector from a quantum circuit.
# Args:
#   - `qc`: Quantum circuit.
# Returns:
#   - State vector of the quantum circuit.

def get_state(qc):
    # Specify the backend as the statevector simulator
    backend = BasicAer.get_backend('statevector_simulator')
    
    # Define the number of shots for the simulation
    numOfShots = 1048576
    
    # Execute the quantum circuit and retrieve the result
    job = execute(qc, backend, shots=numOfShots)
    result = job.result()
    
    # Get the state vector from the result
    state = result.get_statevector()
    
    return state

# The `get_counts` function obtains counts from a quantum circuit.
# Args:
#   - `qc`: Quantum circuit.
#   - `qubit`: Number of qubits.
#   - `shots`: Number of shots (optional, defaults to 2048).
# Returns:
#   - Quantum counts.

def get_counts(qc, qubit, shots=None):
    # Create a list of control qubits 'n' to measure
    n = [i for i in range(qubit)]
    qc.measure(n, n)
    
    # Specify the backend as the QASM simulator
    backend_sim = Aer.get_backend('qasm_simulator')
    
    # Define the number of shots for the simulation (default: 2048 if not specified)
    if shots is None:
        numOfShots = 2**(2*qubit-2)
    else:
        numOfShots = shots
    
    # Execute the quantum circuit and retrieve the counts
    job = execute(qc, backend_sim, shots=numOfShots)
    result = job.result()
    counts = result.get_counts(qc)
    
    return counts

# The `make_square_image` function creates a square image from the input image.
# Args:
#   - `image`: PIL Image.
# Returns:
#   - Square PIL Image.

def make_square_image(image):
    # Get the width and height of the input image
    width, height = image.size
    
    # Calculate the size of the square image based on the larger dimension
    max_side = 2 ** math.ceil((math.log(max(width, height), 2)))
    
    # Create a new square image with black background
    square_image = Image.new('RGB', (max_side, max_side), (0, 0, 0))
    
    # Calculate the coefficient for resizing the original image to fit the square
    coef = max_side / max(width, height) - 0.02
    
    # Resize the original image and paste it onto the square image
    image = image.resize((int(width * coef), int(height * coef)))
    square_image.paste(image, (0, 0))
    
    return square_image

# The `image_preparation` function prepares an image for encoding.
# Args:
#   - `image`: Image data (NumPy array or PIL Image).
#   - `num_qubits`: Number of qubits (optional, auto-determined if None).
# Returns:
#   - Prepared PIL Image.

def image_preparation(image, num_qubits=None):
    # Check if the input image data is a NumPy array
    if type(image) == np.ndarray:
        # Convert the NumPy array to a PIL Image
        image = Image.fromarray(image)
    else:
        pass
    
    # Get the dimensions of the image
    a, b = image.size
    
    # If the image is not a square, make it square
    if a != b:
        image = make_square_image(image)
    
    # Convert the image to grayscale
    image = image.convert('L')
    
    # If the number of qubits is not specified, calculate it based on image dimensions
    if num_qubits is None:
        num_qubits = 2 * math.ceil((math.log(max(a, b), 2))) + 1
        pixels = 2 ** (num_qubits - 1)
        picture_side = int(np.sqrt(pixels))
        print("Num of qubits", num_qubits)
        return image
    else:
        # Resize the image to match the specified number of qubits
        pixels = 2 ** (num_qubits - 1)
        picture_side = int(np.sqrt(pixels))
        image = image.resize((picture_side, picture_side))
        print("Image side", picture_side)
        return image

# The `calculate_mse` function calculates the Mean Squared Error between two images.
# Args:
#   - `image1`: First image (PIL Image).
#   - `image2`: Second image (PIL Image).
# Returns:
#   - Mean Squared Error value.

def calculate_mse(image1, image2):
    # Convert the PIL Images to NumPy arrays
    img1_array = np.array(image1)
    img2_array = np.array(image2)

    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(img1_array, img2_array)

    return mse

# The `calculate_ssim` function calculates the Structural Similarity Index between two images.
# Args:
#   - `image1`: First image (PIL Image).
#   - `image2`: Second image (PIL Image).
# Returns:
#   - Structural Similarity Index value.

def calculate_ssim(image1, image2):
    # Convert the PIL Images to NumPy arrays
    img1_array = np.array(image1)
    img2_array = np.array(image2)
    
    # Calculate the Structural Similarity Index (SSIM)
    ssim = structural_similarity(img1_array, img2_array, multichannel=False, data_range=255)

    return ssim
