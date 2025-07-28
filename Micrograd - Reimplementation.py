# Lessons Learnt
# 
# Using __repr__ to adjust the print of an object
# Modifying python's default operator functions to work for a class object
# Setting a function as an attribute
# _backward vs _backward() for a function
# Setting the attribute of another object as a function
# Defining a function using closure which retains references to variables
# Using isinstance to convert float/int to a class object when using python operator
# Using the reverse of python operator
# need to update in the opposite direction to the gradient of Loss wrt to parameter
# Only update learnable parameters
# Rerun loss.backward after each weight update
# XOR dataset needs two hidden neurons
# Neuron holds all the weights, inputs are not neurons 

import math
import random

class Value():

    def __init__(self, data, children=()):

        self.data = data 
        self.grad = 0.0
        self._backward = lambda:None
        self._children = children 

    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, children=(self, other))

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
 
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, children=(self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError('Only allow to do power with ints/floats')

        out = Value(self.data ** other, children=(self,))

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad            

        out._backward = _backward

        return out

    def __sub__(self, other):
        return self + (-1.0*other)
    
    def __neg__(self):
        return -1.0*self
    
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return other + (-1.0*self)
    
    def tanh(self):

        t = (math.exp(self.data) - math.exp(-self.data)) / (math.exp(self.data) + math.exp(-self.data))

        out = Value(t, children=(self, ))

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        
        out._backward = _backward
        return out

    def sigmoid(self):

        t = 1 / (1 + math.exp(-self.data))

        out = Value(t, children=(self,)) 
        
        def _backward():
            self.grad += t * (1 - t) * out.grad
        
        out._backward = _backward

        return out
    
    def log(self):

        epsilon = 1e-9 # A very small number

        clamped_data = max(self.data, epsilon)
        clamped_data = min(clamped_data, 1.0 - epsilon)

        out_data = math.log(clamped_data)

        out = Value(out_data, children=(self, ))

        def _backward():
            self.grad += ( 1 / clamped_data) * out.grad
        
        out._backward = _backward

        return out

    def backward(self):
        """ Updates all the gradients """

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

class Neuron():
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] 
        self.b = Value(random.uniform(-1, 1)) 
        
    def parameters(self):
        return [self.b] + self.w
    
    def __call__(self, x):
        sum = 0
        for w_i, x_i in zip(self.w, x):
            sum += w_i * x_i
        sum += self.b

        return sum.tanh()

class Layer():

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)] 
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
    
    def __call__(self, x):
        sum = []
        for neuron in self.neurons:
            sum.append(neuron(x))
            
        return sum 

class MLP():
    def __init__(self, nin, nout):
        size = [nin] + nout 
        self.layers = []
        for i in range(len(size) - 1):
            self.layers.append(Layer(size[i], size[i+1]))  
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
        
    def __call__(self, x):
        x = x if isinstance(x, list) else [x] # incase only one input

        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x 


if '__main__' == __name__:
    
    # Simple function example
    def simple_function_example():
        w = Value(4.0)
        x = Value(2.0)
        b = Value(3.0)
        m = Value(6.0)
        print(w)

        # c = w*x
        c = (w*x + b) * m
        c.grad = 1.0
        print(c)

        c.backward()
        print('m grad: ', m.grad)
        print('w grad: ', w.grad)
        print('x grad: ', x.grad)
        print('b grad: ', b.grad)

        print('-------')
        print()
        print()
        print()

    # simple_function_example()

    # Loss function example
    def loss_function_example():
        m = Value(0.0)
        x = Value(1.0)
        b = Value(0.0)

        y_target = 1
        y_hat = m*x + b
        loss = 0.5*(y_hat - y_target)**2

        LR = 0.01

        print('Initial State:')
        print('m is: ', m)
        print('x is: ', x)
        print('b is: ', b)
        print('y_target is: ', y_target)
        print('loss: ', loss)
        print('')

        epochs = 1
        for epoch in range(epochs):

            y_hat = m*x + b
            loss = 0.5*(y_hat - y_target)**2

            m.grad = 0.0
            b.grad = 0.0
            x.grad = 0.0

            loss.backward()

            m.data += - LR * m.grad 
            b.data += - LR * b.grad 

            print(f"Epoch {epoch + 1}")
            print('m is: ', m)
            print('x is: ', x)
            print('b is: ', b)
            print('y_hat is: ', y_hat)
            print('loss: ', loss)
            print('')

# loss_function_example()

# Tiny Dataset Example (Linear & SGD)
def tiny_dataset_linear_SGD():
    x_data = [0, 1, 2]
    y_targets_data = [0, 1, 2]

    epochs = 2
    LR = 0.1

    m = Value(0.0)
    b = Value(0.0)

    print("--- Initial Model State ---")
    print(f"m: {m}")
    print(f"b: {b}")
    print("-" * 30)

    for epoch in range(epochs):

        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        epoch_loss_sum = 0.0


        for i, (x_i, y_i) in enumerate(zip(x_data, y_targets_data)):

            x = Value(x_i)            
            y_target = Value(y_i)

            y_hat = m * x + b
            loss = 0.5 * (y_hat - y_target) ** 2

            epoch_loss_sum += loss.data            

            # Zero Gradients of Learnable Parameters
            m.grad = 0.0
            b.grad = 0.0

            # Compute grad Loss wrt Parameters
            loss.backward()

            print(f"  Example {i + 1} (x={x_i}, y_target={y_i}):")
            print(f"    y_hat (data): {y_hat.data:.4f}")
            print(f"    Loss (data): {loss.data:.4f}")
            print(f"    m.grad (before update): {m.grad:.4f}")
            print(f"    b.grad (before update): {b.grad:.4f}")

            # Update Parameters
            m.data +=  - LR * m.grad
            b.data +=  - LR * b.grad

            print(f"    m.data (after update): {m.data:.4f}")
            print(f"    b.data (after update): {b.data:.4f}")
            print("-" * 20) # Separator for each example

        avg_epoch_loss = epoch_loss_sum / len(x_data)
        print(f"  --- End of Epoch {epoch + 1} Summary ---")
        print(f"  Average Epoch Loss: {avg_epoch_loss:.4f}")
        print(f"  Current m: {m.data:.4f}")
        print(f"  Current b: {b.data:.4f}")
        print("-" * 30)

# tiny_dataset_linear_SGD()

# Tiny Dataset Example (Linear & BGD)
def tiny_dataset_linear_BGD():
    x_data = [0, 1, 2]
    y_targets_data = [0, 1, 2]

    epochs = 2
    LR = 0.1

    m = Value(0.0)
    b = Value(0.0)

    print("--- Initial Model State ---")
    print(f"m: {m}")
    print(f"b: {b}")
    print("-" * 30)

    for epoch in range(epochs):

        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        epoch_loss_sum = 0.0


        # Zero Gradients of Learnable Parameters
        m.grad = 0.0
        b.grad = 0.0

        for i, (x_i, y_i) in enumerate(zip(x_data, y_targets_data)):

            x = Value(x_i)            
            y_target = Value(y_i)

            y_hat = m * x + b
            loss = 0.5 * (y_hat - y_target) ** 2

            epoch_loss_sum += loss.data            

            # Compute grad Loss wrt Parameters
            loss.backward()

            print(f"  Example {i + 1} (x={x_i}, y_target={y_i}):")
            print(f"    y_hat (data): {y_hat.data:.4f}")
            print(f"    Loss (data): {loss.data:.4f}")
            print(f"    m.grad (before update): {m.grad:.4f}")
            print(f"    b.grad (before update): {b.grad:.4f}")

        # Update Parameters
        m.data +=  - LR * m.grad
        b.data +=  - LR * b.grad

        avg_epoch_loss = epoch_loss_sum / len(x_data)
        print(f"  --- End of Epoch {epoch + 1} Summary ---")
        print(f"  Average Epoch Loss: {avg_epoch_loss:.4f}")
        print(f"  Current m: {m.data:.4f}")
        print(f"  Current b: {b.data:.4f}")
        print("-" * 30)

# tiny_dataset_linear_BGD()

# Tiny Dataset Example (Non-Linear & SGD)
def tiny_dataset_nonlinear_BGD():

    # XOR
    a_data = [0.0, 1.0, 0.0, 1.0]
    b_data = [0.0, 0.0, 1.0, 1.0]
    y_targets_data = [0.0, 1.0, 1.0, 0.0]
    
    w1 = Value(random.uniform(-1, 1))
    w2 = Value(random.uniform(-1, 1))
    b1 = Value(random.uniform(-1, 1))

    w3 = Value(random.uniform(-1, 1))
    w4 = Value(random.uniform(-1, 1))
    b2 = Value(random.uniform(-1, 1))

    w5 = Value(random.uniform(-1, 1))
    w6 = Value(random.uniform(-1, 1))
    b3 = Value(random.uniform(-1, 1))

    parameters = [w1, w2, w3, w4, w5, w6, b1, b2, b3]

    epochs = 1000
    LR = 0.2

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}") 

        # Zero Gradients
        for p in parameters:
            p.grad = 0.0

        for i, (a_i, b_i, y_i) in enumerate(zip(a_data, b_data, y_targets_data)):

            h1 = ((w1 * a_i) + (w2 * b_i) + b1).tanh()
            h2 = ((w3 * a_i) + (w4 * b_i) + b2).tanh()
            y_hat = ((w5 * h1) + (w6 * h2) + b3).sigmoid()

            loss = - (y_i * (y_hat).log() + (1 - y_i) * (1 - y_hat).log())
            # loss = 0.5 * (y_hat - y_i) ** 2

            loss.backward()

            print('y_hat: ', y_hat.data, ' y_target: ', y_i)
            print('loss: ', loss)

        # Update
        for p in parameters:
            print(f"    {p}: grad={p.grad:.6f}")
            # print(p)
            p.data += - LR * p.grad

# tiny_dataset_nonlinear_BGD()

def tiny_dataset_using_MLP():
    x_data = [0, 1]
    y_targets_data = [0, 1] 

    model = MLP(1, [2, 1])

    epochs = 1000
    LR = 0.05

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        epoch_loss = 0
        for x_i, y_i in zip(x_data, y_targets_data):

            y_hats = model(x_i)

            loss = 0.5 * (y_hats - y_i)**2
            epoch_loss += loss.data

            # Zero Grad
            for p in model.parameters():
                p.grad = 0.0
            
            loss.backward()

            # Update
            for p in model.parameters():
                p.data += -LR * p.grad 
            
        
            print('New Predictions: ', model(x_i).data)

        print('Loss: ', epoch_loss)
        print('')

tiny_dataset_using_MLP()

