
# Step one - train a large network
# We will take VGG16, drop the fully connected layers, and add three new fully connected layers. We will freeze the convolutional layers, and retrain only the new fully connected layers. In PyTorch, the new layers look like this:
self.classifier = nn.Sequential(
		nn.Dropout(),
		nn.Linear(25088, 4096),
		nn.ReLU(inplace=True),
		nn.Dropout(),
		nn.Linear(4096, 4096),
		nn.ReLU(inplace=True),
		nn.Linear(4096, 2))


# Step two - Rank the filters
# To compute the Taylor criteria, we need to perform a Forward+Backward pass on our dataset (or on a smaller part of it if it’s too large. but since we have only 2000 images lets use that).
# Now we need to somehow get both the gradients and the activations for convolutional layers. In PyTorch we can register a hook on the gradient computation, so a callback is called when they are ready:
for layer, (name, module) in enumerate(self.model.features._modules.items()):
	x = module(x)
	if isinstance(module, torch.nn.modules.conv.Conv2d):
		x.register_hook(self.compute_rank)
		self.activations.append(x)
		self.activation_to_layer[activation_index] = layer
		activation_index += 1

# Now we have the activations in self.activations, and when a gradient is ready, compute_rank will be called:
def compute_rank(self, grad):
	activation_index = len(self.activations) - self.grad_index - 1
	activation = self.activations[activation_index]
	values = \
		torch.sum((activation * grad), dim = 0).\
			sum(dim=2).sum(dim=3)[0, :, 0, 0].data
	
	# Normalize the rank by the filter dimensions
	values = \
		values / (activation.size(0) * activation.size(2) * activation.size(3))

	if activation_index not in self.filter_ranks:
		self.filter_ranks[activation_index] = \
			torch.FloatTensor(activation.size(1)).zero_().cuda()

	self.filter_ranks[activation_index] += values
	self.grad_index += 1

