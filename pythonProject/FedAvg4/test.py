import matplotlib.pyplot as plt

# Data for plotting
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

# Create a plot
plt.plot(x, y, label='y = x^2', marker='o')  # Plot with markers

# Add labels and title
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Simple Plot Example')

# Add a legend
plt.legend()

# Show the plot
plt.show()
