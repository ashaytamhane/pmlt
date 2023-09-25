import torch.nn as nn 
import torch

class MLP(nn.Module):
	def __init__(self,num_inputs,num_hidden,num_outputs,hidden_activation_func=nn.Sigmoid(), output_activation_func=nn.Sigmoid()):
		super().__init__()
		self.input_layer=nn.Flatten() # flatten input
		self.hidden_layer=nn.LazyLinear(num_hidden) # avoid calculating number of inputs for hidden layer
		self.hidden_activation_func=hidden_activation_func
		self.output_layer=nn.Linear(num_hidden,num_outputs)
		self.output_activation_func=output_activation_func

	def forward(self,X):
		inputs=self.input_layer(X)
		hidden_output=self.hidden_activation_func(self.hidden_layer(inputs))
		outputs=self.output_activation_func(self.output_layer(hidden_output))
		return outputs