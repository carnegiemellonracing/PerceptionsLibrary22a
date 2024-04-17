import csv
import matplotlib.pyplot as plt
import numpy as np

COUNTER = 23
def update_plot():
    plt.clf()  # Clear the previous plot
    with open(f'scripts/data/blue_cones/{COUNTER}.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        blue_data = list(csv_reader)

    with open(f'scripts/data/yellow_cones/{COUNTER}.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        yellow_data = list(csv_reader)
    
    with open(f'scripts/data/orange_cones/{COUNTER}.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        orange_data = list(csv_reader)

    # Extract x, y coordinates from the data
    blue_x = [float(row[0]) for row in blue_data]
    blue_y = [float(row[1]) for row in blue_data]

    yellow_x = [float(row[0]) for row in yellow_data]
    yellow_y = [float(row[1]) for row in yellow_data]

    orange_x = [float(row[0]) for row in orange_data]
    orange_y = [float(row[1]) for row in orange_data]

    blue_coefficients = np.polyfit(blue_x, blue_y, 3)
    yellow_coefficients = np.polyfit(yellow_x, yellow_y, 3)

    blue_fit_x = np.linspace(min(blue_x), max(blue_x), 1000)
    blue_fit_y = np.polyval(blue_coefficients, blue_fit_x)

    yellow_fit_x = np.linspace(min(yellow_x), max(yellow_x), 1000)
    yellow_fit_y = np.polyval(yellow_coefficients, yellow_fit_x)

    # Plot the cones
    plt.scatter(blue_fit_x, blue_fit_y, color='black', label=f'BLUE SPLINE')
    plt.scatter(yellow_fit_x, yellow_fit_y, color='black', label=f'YELLOW SPLINE')
    plt.scatter(blue_x, blue_y, color='blue', label=f"Blue Cones")
    plt.scatter(yellow_x, yellow_y, color='orange', label=f"Yellow Cones")
    plt.scatter(orange_x, orange_y, color='green', label=f"Orange Cones")
    plt.xlim((-10, 10))
    plt.ylim((0, 20))
    plt.legend()
    plt.draw()

def on_key(event):
    global COUNTER
    if event.key == 'left':
        COUNTER = COUNTER - 1
    elif event.key == 'right':
        COUNTER = COUNTER + 1

    update_plot()    

# Create a figure and axis
fig, ax = plt.subplots()

# Connect the key press event to the function on_key
plt.connect('key_press_event', on_key)

# Initial plot
update_plot()

# Display the Matplotlib window and wait for user input
plt.show()
plt.pause(0.01)  # Ensure the plot is rendered before waiting for input

# Wait for user input to exit (close the window manually)
while True:
    plt.pause(1)
