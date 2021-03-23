import torch.nn as nn

class MultipleRegression(nn.Module):
    def __init__(self, num_features, num_neurons):
        super(MultipleRegression, self).__init__()
        self.layer_1 = nn.Linear(num_features, num_neurons)
        self.layer_2 = nn.Linear(num_neurons, num_neurons)
        self.layer_3 = nn.Linear(num_neurons, num_neurons)
        self.layer_4 = nn.Linear(num_neurons, num_neurons)
        self.layer_5 = nn.Linear(num_neurons, num_neurons)
        self.layer_out = nn.Linear(num_neurons, 1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.relu(self.layer_2(x))
            x = self.relu(self.layer_3(x))
            x = self.relu(self.layer_4(x))
            x = self.relu(self.layer_5(x))
            x = self.layer_out(x)
            return (x)
    
    def predict(self, test_inputs):
            x = self.relu(self.layer_1(test_inputs))
            x = self.relu(self.layer_2(x))
            x = self.relu(self.layer_3(x))
            x = self.relu(self.layer_4(x))
            x = self.relu(self.layer_5(x))
            x = self.layer_out(x)
            return (x)