import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from matplotlib.animation import FuncAnimation

def update_wave_function(k1, w1, A1, k2, w2, A2, line, x, dx, t):
    """
    Updates the wave function based on wave numbers, angular speeds, amplitudes, time, and x-coordinates.

    Parameters:
        k1 (float): Wave number of the first harmonic.
        w1 (float): Angular speed of the first harmonic.
        A1 (float): Amplitude of the first harmonic.
        k2 (float): Wave number of the second harmonic.
        w2 (float): Angular speed of the second harmonic.
        A2 (float): Amplitude of the second harmonic.
        line (Line2D): Line object representing the plot.
        x (array-like): Array of x-coordinates.
        t (float): Time value.
        dx (float): The length that x(y1) speeds up x(y1)
    """
    y1 = A1 * np.cos(k1 * (x + dx) - w1 * t)
    y2 = A2 * np.cos(k2 * x - w2 * t)
    y = y1 + y2
    line.set_ydata(y)

def plot_harmonic_wave(k1, w1, A1, k2, w2, A2, dx):
    """
    Plots a propagating harmonic wave function using Matplotlib.

    Parameters:
        k1 (float): Wave number of the first harmonic.
        w1 (float): Angular speed of the first harmonic.
        A1 (float): Amplitude of the first harmonic.
        k2 (float): Wave number of the second harmonic.
        w2 (float): Angular speed of the second harmonic.
        A2 (float): Amplitude of the second harmonic.
    """
    # Set k, w max values
    max = 100

    # Generate x-coordinates from -5 to 5 with 0.1 increments
    x = np.arange(-5, 5, 0.05)

    # Create the figure and axes with a larger figsize
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.3)

    # Set labels and title
    ax.set_xlabel('distance [m]')
    ax.set_ylabel('Amplitude [m]')
    ax.set_title('Propagating Harmonic Wave Function')

    # Initialize wave function
    y1 = A1 * np.cos(k1 * (x + dx))
    y2 = A2 * np.cos(k2 * x)
    y = y1 + y2

    # Create the line plot
    line, = ax.plot(x, y)

    # Add sliders for wave numbers, angular speeds, and amplitudes
    ax_k1 = plt.axes([0.15, 0.18, 0.65, 0.03])
    ax_w1 = plt.axes([0.15, 0.15, 0.65, 0.03])
    ax_A1 = plt.axes([0.15, 0.12, 0.65, 0.03])
    ax_k2 = plt.axes([0.15, 0.09, 0.65, 0.03])
    ax_w2 = plt.axes([0.15, 0.06, 0.65, 0.03])
    ax_A2 = plt.axes([0.15, 0.03, 0.65, 0.03])
    ax_dx = plt.axes([0.15, 0.21, 0.65, 0.03])

    slider_k1 = Slider(ax_k1, 'Wave number 1 (k1)', -max, max, valinit=k1, valstep=0.1)
    slider_w1 = Slider(ax_w1, 'Angular speed 1 (w1)', -max, max, valinit=w1, valstep=0.1)
    slider_A1 = Slider(ax_A1, 'Amplitude 1 (A1)', 0, 5, valinit=A1, valstep=0.1)
    slider_k2 = Slider(ax_k2, 'Wave number 2 (k2)',  -max, max, valinit=k2, valstep=0.1)
    slider_w2 = Slider(ax_w2, 'Angular speed 2 (w2)', -max, max, valinit=w2, valstep=0.1)
    slider_A2 = Slider(ax_A2, 'Amplitude 2 (A2)', 0, 5, valinit=A2, valstep=0.1)
    slider_dx = Slider(ax_dx, 'd. along the path (dx)', -max, max, valinit=dx, valstep=0.1)


    # Function to update the plot when sliders change
    def update(val):
        nonlocal k1, w1, A1, k2, w2, A2, dx
        k1 = slider_k1.val
        w1 = slider_w1.val
        A1 = slider_A1.val
        k2 = slider_k2.val
        w2 = slider_w2.val
        A2 = slider_A2.val
        dx = slider_dx.val

    # Register the update function with the sliders
    slider_k1.on_changed(update)
    slider_w1.on_changed(update)
    slider_A1.on_changed(update)
    slider_k2.on_changed(update)
    slider_w2.on_changed(update)
    slider_A2.on_changed(update)
    slider_dx.on_changed(update)

    # Add text boxes for wave numbers, angular speeds, and amplitudes
    box_k1 = plt.axes([0.90, 0.18, 0.08, 0.03])
    box_w1 = plt.axes([0.90, 0.15, 0.08, 0.03])
    box_A1 = plt.axes([0.90, 0.12, 0.08, 0.03])
    box_k2 = plt.axes([0.90, 0.09, 0.08, 0.03])
    box_w2 = plt.axes([0.90, 0.06, 0.08, 0.03])
    box_A2 = plt.axes([0.90, 0.03, 0.08, 0.03])
    box_dx = plt.axes([0.90, 0.21, 0.08, 0.03])

    text_box_k1 = TextBox(box_k1, 'k1', initial=str(k1))
    text_box_w1 = TextBox(box_w1, 'w1', initial=str(w1))
    text_box_A1 = TextBox(box_A1, 'A1', initial=str(A1))
    text_box_k2 = TextBox(box_k2, 'k2', initial=str(k2))
    text_box_w2 = TextBox(box_w2, 'w2', initial=str(w2))
    text_box_A2 = TextBox(box_A2, 'A2', initial=str(A2))
    text_box_dx = TextBox(box_dx, 'dx', initial=str(dx))

    # Function to update the plot when text boxes change
    def submit_k1(text):
        nonlocal k1
        k1 = float(text)
        slider_k1.set_val(k1)

    def submit_w1(text):
        nonlocal w1
        w1 = float(text)
        slider_w1.set_val(w1)

    def submit_A1(text):
        nonlocal A1
        A1 = float(text)
        slider_A1.set_val(A1)

    def submit_k2(text):
        nonlocal k2
        k2 = float(text)
        slider_k2.set_val(k2)

    def submit_w2(text):
        nonlocal w2
        w2 = float(text)
        slider_w2.set_val(w2)

    def submit_A2(text):
        nonlocal A2
        A2 = float(text)
        slider_A2.set_val(A2)

    def submit_dx(text):
        nonlocal dx
        dx = float(text)
        slider_dx.set_val(dx)

    text_box_k1.on_submit(submit_k1)
    text_box_w1.on_submit(submit_w1)
    text_box_A1.on_submit(submit_A1)
    text_box_k2.on_submit(submit_k2)
    text_box_w2.on_submit(submit_w2)
    text_box_A2.on_submit(submit_A2)
    text_box_dx.on_submit(submit_dx)

    # Animation function
    def animate(frame):
        t = frame / 40.0  # Time update per frame
        update_wave_function(k1, w1, A1, k2, w2, A2, line, x, dx, t)
        ax.set_ylim(-A1 - A2 - 0.2, A1 + A2 + 0.2)
        ax.grid(True)

    # Create animation
    animation = FuncAnimation(fig, animate, frames=1000, interval=10, repeat=True)
    ax.set_ylim(-A1-A2-0.2, A1+A2+0.2)

    plt.show()


# Parameters for the wave propagation
wave_number1 = 0
angular_speed1 = 1.0
amplitude1 = 1.0
wave_number2 = 0
angular_speed2 = 1.0
amplitude2 = 1.0
dx = 0

# Plot the propagating harmonic wave function with sliders
plot_harmonic_wave(wave_number1, angular_speed1, amplitude1, wave_number2, angular_speed2, amplitude2, dx)


