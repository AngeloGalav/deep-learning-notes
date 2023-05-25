## Modeling sequences
Typical problems:
- turn an _input sequence into an output sequence_ (possibly in a different domain):
	- _translation_ between different languages
	- speech/sound recognition
	- ...
- predict the _next term in a sequence_
	- The target output sequence is the input sequence with an advance of 1 step. Blurs the distinction between supervised and unsupervised learning.
- _predict a result from a temporal sequence of states_, Typical of Reinforcement learning, and robotics.

## Memoryless approach
Compute the output as a result of a fixed number of elements in the input sequence:
![[memoryless.png]]
Used e.g. in 
- Bengio’s (first) predictive natural language model 
- Qlearning for Atari Games

## What is a RNN? 
(This is an exercerpt from the final part of the previous lesson)
An RNN is simply a neural network with cycles in it. The end. 
This means that, in presence of backward connections, ==hidden states depend on the _past history_ of the net==, so it has some kind of memory in a sense. 
![[RNN.png]]

As we know, in logical circuits having cycles cause some instabilitites...
![[circuit.png]]
...but these are solved usually by adding a _clock_.
A similar concepts is preserved in RNN thanks to __Temporal Unfolding__, meaning that _activations are updated a precise time steps_.
In this way, the RNN is basically a layered net that keeps reusing the same weights:
![[RNN_2.png]]
So it can easily be translated into a traditional feedforward NN. The only thing that we have to keep in mind is that the weights are shared between weights of the same layer at the start; however, they get updated differently after the first update.  

#### Sharing weights through time 
It is easy to modify the backprop algorithm to _incorporate equality constraints_ between weights. We compute the gradients as usual, and then _average gradients_ so that they induce a _same update_ (and preserve the weights). 
- If the initial weights satisfied the constraints from the start, they will continue to do.
- N.B.: this same update is done if we want to preserve the same weights. 

To constrain $w_1 = w_2$ we need:
- $∆w_1 = ∆w_2$
- compute $\dfrac{∂E}{∂w_1}$ and $\dfrac{∂E}{∂w_2}$ and use $\dfrac{∂E}{∂w_1} + \dfrac{∂E}{ ∂w_2}$ to update both $w_1$ and $w_2$. 

#### Backpropagation through time - BPTT
- think of the recurrent net as _a layered, feed-forward net_ with _shared weights_ and train the feed-forward net _with weight constraints_. 
- reasoning in the time domain: 
	- the forward pass builds up a stack of the activities of all the units at each time step.
	- the _backward pass_ peels activities _off the stack_ to compute the error derivatives at each time step. 
	- finally we add together the derivatives at all the different times for each weight.

#### Hidden state initialization
We need to specify the initial activity state of all the _hidden_ and _output units_. The best approach is to treat them as parameters, _learning them in the same way as we learn the weights_: 
- start off with an initial random guess for the initial states. 
- at the end of each training sequence, backpropagate through time all the way to the initial states to get the gradient of the error function with respect to each initial state. 
- adjust the initial states by following the negative gradient. 

## Long-Short Term Memory (LSTM)
![[lstm1.png]]
Both the vector of inputs and the vector of outputs have the same length $t$.

### A simple, traditional RNN
Let's see another example.
The content of the memory cell $C_t$ , and the input $x_t$ are combined through a simple neural net to produce the output $h_t$ that _coincides with the new content_ of the cell $C_{t+1}$.
![[RNN_yo.png]]
Why $C_{t+1} = h_t$? Better trying to _preserve the memory cell_, letting the neural net _learn how and when to update it_.
- Many times, though, using these kind of the structure the memory may be lost in a way (because of the input $x_t$). Nevertheless, we try to preserve the memory as much as possible.  

Also, $C_t \not = h_t$, since the content of a cell, before becoming the output, goes through some kind of post processing.  

### The overall structure of a LSTM 
![[lstm_arch.png]]

#### C-line and gates 
The LSTM has the ability _to remove or add information to __the cell state___, in a way regulated by suitable __gates__. 
__Gates__ are a way to optionally let information through: _the product with a sigmoid neural net layer simulates a __boolean mask___.
![[cline_gate.png]]

#### The forget gate 
The __forget gate__ decides what part of the memory cell to _preserve_.
![[ok_forget_me_too.png]]
In particular, by concatenating the input of the current cell w/ the output of the previous, which is then passed to a network layer, it generates a _mask_ which decides which part of the content of the previous to keep and which of them to ignore.  
This is a form of _attention_, as we will see. 

#### The update gate 
![[update_gate.png]]
As we've seen, the _input gate_ decides _what part of the input to preserve_. 
The $tanh$ layer creates a vector of new candidate values $\tilde C_t$ to be added to the state.
Here's how the content of a cell is updated. 
![[cell_update.png]]
We multiply the old state by the boolean mask $f_t$ . Then we add $i_t ∗ \tilde C_t$.

#### The output gate
The output $h_t$ is a filtered version of the content of the cell.
![[output_gate.png]]
The output gate decides what parts of the cell state to output. The $tanh$ function is _used to __renormalize__ values_ in the interval [−1, 1].

## Applications 
They were used for NLP until the birth of transformers.


Asperti then did a long ass demo. You can find the demo [here](https://virtuale.unibo.it/pluginfile.php/1623565/mod_resource/content/1/carry_over.ipynb).

-----
