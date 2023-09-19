import torch
import networkx as nx
import matplotlib.pyplot as plt

# Define a simple 3-layer neural network
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.fc2 = torch.nn.Linear(8, 6)
        self.fc3 = torch.nn.Linear(6, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the neural network
model = SimpleNN()

# Access the weights of the first layer
weights1 = model.fc1.weight.data.numpy()
weights2 = model.fc2.weight.data.numpy()
weights3 = model.fc3.weight.data.numpy()

# Create a graph
G = nx.Graph()

# Add input, hidden, and output nodes to the graph
for i in range(4):
    G.add_node(f'Input {i}')

for i in range(8):
    G.add_node(f'Hidden 1 {i}')

for i in range(6):
    G.add_node(f'Hidden 2 {i}')

for i in range(3):
    G.add_node(f'Output {i}')

# Add edges based on the weights for each layer
def add_edges(weights, layer_name):
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weight_value = weights[i, j]
            G.add_edge(f'Input {j}', f'{layer_name} {i}', weight=weight_value)

add_edges(weights1, 'Hidden 1')
add_edges(weights2, 'Hidden 2')
add_edges(weights3, 'Output')

# Create a layout for the nodes (using shell_layout for vertical alignment)
layout = {}
layout.update((node, (layer_index, node_index * 50))
              for layer_index, layer_name in enumerate(['Input', 'Hidden 1', 'Hidden 2', 'Output'])
              for node_index, node in enumerate(G.nodes()) if node.startswith(layer_name))
pos = layout

# Draw the graph with weights as labels
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('3-Layer Neural Network Weight Visualization (Vertical Alignment)')
plt.show()
