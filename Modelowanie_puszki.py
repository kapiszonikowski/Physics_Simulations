import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

########################################################################################################################

# Initial values
pi = np.pi
u = 4 * pi * 10**(-7)                   # Przenikalność magnetyczna próżni
n = 40                                  # Numer kolejnego pkt przybliżenia liniowego
i = 100000                              # Ilość punktów opisujących czas
t_max = 2*10**(-4)                      # Końcowy czas obliczeń
h = t_max / i                           # Przeskok dla numerycznego całkowania
print(h)
V0 = 4.9 * 10**3                        # Napięcie początkowe
R = 0.35                                # Opór układu
N = 3                                   # Ilość zwojów cewki
r_coil = 0.0345                         # Promień cewki
r_can = r_coil - 0.0015
C = 210 * 10**(-6)                      # Pojemność kondensatorów
d_coil_wire = 0.002                     # Średnica drutu do konstrukcji cewki
ro_al = 2.82 * 10**(-8)                 # Oporność właściwa aluminium
Can_Wall_Thickness = 0.097 * 0.001
S_can = pi * r_can ** 2                 # Powierzchnia przekroju puszki

t = np.linspace(0, t_max, i)            # Time array

Can_Conductance_distance = 0.05
Coil_Perimeter = 2 * pi * r_coil
Can_Perimeter = 2 * pi * r_can
Can_Resistance = (ro_al * Can_Perimeter) / (Can_Conductance_distance * Can_Wall_Thickness)

########################################################################################################################
def Trapezoid_Integration(func, h):
    Integrated_Value = func[0] * h / 2
    for a in range(1, i - 1):
        Integrated_Value += func[a] * h
    Integrated_Value += func[-1] * h / 2
    return Integrated_Value

def Integrative_calculation(f1, f2, h):
    Value = [a for a in range(i-1)]
    Value[0] = f1[0] * f2[0] * h / 2
    for a in range(1, i - 2):
        Value[a] = Value[a-1] + f1[a] * f2[a] * h
    Value[i-1] = Value[i-2] + f1[i-1] * f2[i-1] * h / 2
    return Value

########################################################################################################################

fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(12, 10))
plt.subplots_adjust(bottom=0.19, hspace=0.34, wspace=0.3, right=0.964, left=0.065, top=0.966)

ax_slider_R = plt.axes([0.15, 0.1, 0.65, 0.03])
slider_R = Slider(ax_slider_R, 'Resistance (R)', 0.001, 5, valinit=R, valstep=0.0001)

ax_slider_N = plt.axes([0.15, 0.06, 0.65, 0.03])
slider_N = Slider(ax_slider_N, 'Turns (N)', 1, 7, valinit=N, valstep=1)

ax_slider_r_coil = plt.axes([0.15, 0.02, 0.65, 0.03])
slider_r_coil = Slider(ax_slider_r_coil, 'r_coil', r_coil - 10 * 0.0065, r_coil, valinit=r_coil, valstep=0.0065)

########################################################################################################################

# Initial Calculations
L = N * u * pi * r_coil ** 2 / d_coil_wire
L_can = u * r_can ** 2 / Can_Conductance_distance
B = R / (2 * L)
w0 = (1 / (L * C)) ** (1/2)
if B > w0:
    w = np.sqrt(B**2 - w0**2)
    Coil_Current = V0 / (2 * L * w) * np.exp(-B * t) * (np.exp(w * t) - np.exp(-w * t))
    """CC_der = np.gradient(Coil_Current, t)
    Electromotive_Power_Inside_The_Can = - u * S_can / Can_Conductance_distance * CC_der
    Integral_CC_der_multip_by_exp_R_over_L = 0"""

elif w0 > B:
    w1 = np.sqrt(w0**2 - B**2)
    Coil_Current = V0 / (L * w1) * np.exp(-B * t) * np.cos(w1 * t)
    """CC_der = - V0 / (L * w1) * np.exp(-B * t) * (w1 * np.sin(w1 * t) + B * np.cos(w1 * t))
    Electromotive_Power_Inside_The_Can = - u * S_can / Can_Conductance_distance * CC_der
    Integral_CC_der_multip_by_exp_R_over_L = V0 / (L * w1) * (- u * S_can / Can_Conductance_distance) * (B * (np.exp(
        R/L) - B) * w1 * np.sin(w1 * t) + (-B**2*np.exp(R/L) - w1**2) * np.cos(w1 * t))"""

else:
    Coil_Current = V0 / L * t * np.exp(-B * t)
    """CC_der = - V0 / L * (B * t - 1) * np.exp(-B * t)
    Electromotive_Power_Inside_The_Can = - u * S_can / Can_Conductance_distance * CC_der
    Integral_CC_der_multip_by_exp_R_over_L = 0"""

# Printing Initial values
print('')
print(f"Inductance L: {L}")
print(f"Coefficient of resistance B: {B}")
print(f"Angular frequency w0: {w0}")
try:
    print(f"Angular frequency w: {w}")
except:
    print(f"Angular frequency w: {w1}")

print(f"Can resistance: {Can_Resistance}")
print('')

# Real Formulas
Coil_Voltage = L * np.gradient(Coil_Current, t)
Magnetic_Potential_B = u * Coil_Current / d_coil_wire
Magnetic_Flux_Coil = Magnetic_Potential_B * S_can
Electromotive_Power_Inside_The_Can = - np.gradient(Magnetic_Flux_Coil, t)
Variable_parameter = Electromotive_Power_Inside_The_Can * np.exp(Can_Resistance * t / L)        #trzeba poprawić
Can_Current = Trapezoid_Integration(Variable_parameter, h) * np.exp(-Can_Resistance * t / L)    #trzeba poprawić
Forces_on_can_by_length = Can_Current * Magnetic_Flux_Coil
Total_Forces_on_can = Can_Current * Magnetic_Flux_Coil * Can_Perimeter
Watts_in_can = Can_Current * Electromotive_Power_Inside_The_Can


print(f"Magnetic_Potential_B {Magnetic_Potential_B[1]}")
print(f"Electromotive_Power_Inside_The_Can: {Electromotive_Power_Inside_The_Can[1]}")
print(f"Can Current: {Can_Current[1]}")
print(f"Can Magnetic potential: {Can_Current[1] * u / Can_Conductance_distance}")

#print(f"check:  {u * I0 * S_can / d_coil_wire * np.real(np.exp(-B * t[0]) * (B * np.cos(w * t[0] + alpha + fi) + w * np.sin(w * t[0] + alpha + fi)))}")

# Energy Calculations
Joule_Heat_in_can = Trapezoid_Integration(Watts_in_can, h)

print(f"Overall Energy: {C * V0**2 / 2}")
print(f"Joule Heat emitted in can: {Joule_Heat_in_can}")

"""xxx = Integrative_calculation(t,t,h)
print(xxx[-1])"""

########################################################################################################################

# Update function for the slider
def update(val):
    global R, N, r_coil

    # Create a 0 lvl line
    ax1.lines[0].set_ydata(np.zeros(i))
    ax2.lines[0].set_ydata(np.zeros(i))
    ax3.lines[0].set_ydata(np.zeros(i))
    ax4.lines[0].set_ydata(np.zeros(i))
    ax5.lines[0].set_ydata(np.zeros(i))
    ax6.lines[0].set_ydata(np.zeros(i))


    R = slider_R.val
    N = slider_N.val
    r_coil = slider_r_coil.val

    # Initial calculations
    L = N * u * pi * r_coil ** 2 / d_coil_wire
    B = R / (2 * L)
    w0 = (1 / (L * C)) ** (1 / 2)
    if B > w0:
        w = np.sqrt(B ** 2 - w0 ** 2)
        Coil_Current = V0 / (2 * L * w) * np.exp(-B * t) * (np.exp(w * t) - np.exp(-w * t))
    elif w0 > B:
        w1 = np.sqrt(w0 ** 2 - B ** 2)
        Coil_Current = V0 / (L * w1) * np.exp(-B * t) * np.cos(w1 * t)
    else:
        Coil_Current = V0 / L * t * np.exp(-B * t)

    # Formulas
    Coil_Voltage = L * np.gradient(Coil_Current, t)
    Magnetic_Potential_B = u * Coil_Current / d_coil_wire
    Magnetic_Flux_Coil = Magnetic_Potential_B * S_can
    Electromotive_Power_Inside_The_Can = - np.gradient(Magnetic_Flux_Coil, t)   # trzeba zastąpić
    Variable_parameter = Electromotive_Power_Inside_The_Can * np.exp(Can_Resistance * t / L)
    Can_Current = Trapezoid_Integration(Variable_parameter, h) * np.exp(-Can_Resistance * t / L)
    Forces_on_can_by_length = Can_Current * Magnetic_Flux_Coil
    Total_Forces_on_can = Can_Current * Magnetic_Flux_Coil * Can_Perimeter
    Watts_in_can = Can_Current * Electromotive_Power_Inside_The_Can

    Joule_Heat_in_can = Trapezoid_Integration(Watts_in_can, h)

    print(f"Joule Heat emitted in can: {Joule_Heat_in_can}")

    # Plotting new formulas
    ax1.lines[1].set_ydata(Electromotive_Power_Inside_The_Can)
    ax2.lines[1].set_ydata(Can_Current)
    ax3.lines[1].set_ydata(Coil_Current)
    ax4.lines[1].set_ydata(Magnetic_Potential_B)
    ax5.lines[1].set_ydata(Watts_in_can)
    ax6.lines[1].set_ydata(Total_Forces_on_can)

    # Update y-axis limits
    ax1.set_ylim(-0.05 * np.max(Electromotive_Power_Inside_The_Can), 1.1 * np.max(Electromotive_Power_Inside_The_Can))
    ax2.set_ylim(-0.05 * np.max(Can_Current), 1.1 * np.max(Can_Current))
    ax3.set_ylim(-0.05 * np.max(Coil_Current), 1.1 * np.max(Coil_Current))
    ax4.set_ylim(-0.05 * np.max(Magnetic_Potential_B), 1.1 * np.max(Magnetic_Potential_B))
    ax5.set_ylim(-0.05 * np.max(Watts_in_can), 1.1 * np.max(Watts_in_can))
    ax6.set_ylim(-0.05 * np.max(Total_Forces_on_can), 1.1 * np.max(Total_Forces_on_can))

    fig.canvas.draw_idle()

slider_N.on_changed(update)
slider_R.on_changed(update)
slider_r_coil.on_changed(update)

########################################################################################################################

# Specify the desired x-axis tick locations
x_ticks = np.linspace(0, 2*10**(-4), 6)  # Adjust the number of ticks as needed

# Format the tick labels
x_tick_labels = [f"{tick:.2e}" for tick in x_ticks]

# Create a 0 lvl line
ax1.plot(t, np.zeros(i), label='Zero line', color="orange", linewidth=0.5)
ax2.plot(t, np.zeros(i), label='Zero line', color="orange", linewidth=0.5)
ax3.plot(t, np.zeros(i), label='Zero line', color="orange", linewidth=0.5)
ax4.plot(t, np.zeros(i), label='Zero line', color="orange", linewidth=0.5)
ax5.plot(t, np.zeros(i), label='Zero line', color="orange", linewidth=0.5)
ax6.plot(t, np.zeros(i), label='Zero line', color="orange", linewidth=0.5)


# Create the plot
ax1.plot(t, Electromotive_Power_Inside_The_Can, label='Discharge curve')
ax2.plot(t, Can_Current, label='Current inside the can')
ax3.plot(t, Coil_Current, label='Current inside the coil')
ax4.plot(t, Magnetic_Potential_B, label='Magnetic potential created by the coil')
ax5.plot(t, Watts_in_can, label='Watts in can')
ax6.plot(t, Total_Forces_on_can, label='Total forces on can')


# Set the x-axis ticks and labels
ax1.set_xticks(x_ticks)
#ax.set_xticklabels(x_tick_labels, rotation=45)  # Adjust rotation if needed
ax2.set_xticks(x_ticks)
ax3.set_xticks(x_ticks)
ax4.set_xticks(x_ticks)
ax5.set_xticks(x_ticks)
ax6.set_xticks(x_ticks)


# Setting limits
ax1.set_ylim(-0.05 * np.max(Electromotive_Power_Inside_The_Can), 1.1 * np.max(Electromotive_Power_Inside_The_Can))
ax2.set_ylim(-0.05 * np.max(Can_Current), 1.1 * np.max(Can_Current))
ax3.set_ylim(-0.05 * np.max(Coil_Current), 1.1 * np.max(Coil_Current))
ax4.set_ylim(-0.05 * np.max(Magnetic_Potential_B), 1.1 * np.max(Magnetic_Potential_B))
ax5.set_ylim(-0.05 * np.max(Watts_in_can), 1.1 * np.max(Watts_in_can))
ax6.set_ylim(-0.05 * np.max(Total_Forces_on_can), 1.1 * np.max(Total_Forces_on_can))


# Add labels and title
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Voltage (V)')
ax1.set_title('Can')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Current (A)')
ax2.set_title('Can')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Current (A)')
ax3.set_title('Coil')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Magnetic Potential B (T)')
ax4.set_title('Coil')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Watts (W)')
ax5.set_title('Can')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Force (N)')
ax6.set_title('Can')

########################################################################################################################

# Show the plot
plt.show()
