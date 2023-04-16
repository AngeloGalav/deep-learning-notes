---
TODO: 
- rewatch lesson
---
# Expressiveness
This lesson is focused in what we can compute with a NN. 

Suppose we have a single layer NN.
![[perceptron.png]]

For the moment, let's suppose that we have a binary function as an activation function:
![[perceptron_formula.png]]
The bias allow us to _fix_ the _threshold_ that we're _interested_ in. 

## Hyperplane
The set of points:
![[simple_equation.png]]
defines a hyperplane in the space of the variables $x_i$. 
![[line_example.png]]

The _hyperplane_ divides the space in _two parts_: 
- to one of them (above the line) the perceptron gives value 1,
- to the other (below the line) value 0.

### NN logical connections
![[nand.png]]
and the answer is...
![[linear_perceptron_nand.png]]

But we _cant_ represent _every_ circuit with a linear perceptron (i.e. XOR).  

Can we recognize these patterns with a perceptron (aka binary threshold)?
![[pixels_lp.png]]
__No__, each pixel should individually contribute to the classification, that is not the case (more in the next slides). 
So considering more than one pixel at a time it's not a linear task. 

Let us e.g. consider the first pixel, and suppose it is black (the white case is symmetric). 
![[pixels_lp_2.png]]
does this improve our knowledge for the purposes of classification?
No, since we have still the same probability to have a good or a bad example.

##### MNIST Example
Can we address digit recognition with linear tools? (perceptrons, logistic regression, . . . )
When we want to use a linear technique for learning, we have to ask ourself, is each one of the features informative by itself or should consider them in a particular context?![[digits.png]]

## Multi-layer perceptrons
- we know we can _compute nand_ with a perceptron 
- we know that nand is logically complete (i.e. we can compute any connective with nands)
- so: why perceptrons are not complete? 
	- answer: because we need _to compose them_ and consider _Multi-layer perceptrons_. 
![[xor_perceptrons.png]]

So... since shallow networks are already complete, why going for _deep networks_?
With deep nets, the same function may be computed with _less neural units_ (Cohen, et al.)
- _Activation functions_ play an essential role, since they are the only source of nonlinearity, and hence of the expressiveness of NNs.

## Formal expressiveness in the continuous case
What can we say instead of continuos functions?
- approximating functions with logistic neurons
