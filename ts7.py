# -*- coding: utf-8 -*-
"""

"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Definir variable z
z = sp.Symbol('z')

# Funciones de transferencia con potencias negativas
T_a = z**(-3) + z**(-2) + z**(-1) + 1
T_b = z**(-4) + z**(-3) + z**(-2) + z**(-1) + 1
T_c = 1 - z**(-1)
T_d = 1 - z**(-2)

print("Forma original:")
print("a) T(z) =", T_a)
print("b) T(z) =", T_b)
print("c) T(z) =", T_c)
print("d) T(z) =", T_d)

# Normalizar (multiplicar por z^k para eliminar potencias negativas)
T_a_norm = sp.expand(T_a * z**3)
T_b_norm = sp.expand(T_b * z**4)
T_c_norm = sp.expand(T_c * z)
T_d_norm = sp.expand(T_d * z**2)

print("\nForma normalizada:")
print("a) T(z) =", T_a_norm)
print("b) T(z) =", T_b_norm)
print("c) T(z) =", T_c_norm)
print("d) T(z) =", T_d_norm)


# Definir rango de frecuencia
omega = np.linspace(0, np.pi, 500)
z = np.exp(1j * omega)

# Calcular módulo y fase
def mag_phase(T):
    mag = np.abs(T)
    phase = np.angle(T)
    return mag, phase

mag_a, phase_a = mag_phase(T_a)
mag_b, phase_b = mag_phase(T_b)
mag_c, phase_c = mag_phase(T_c)
mag_d, phase_d = mag_phase(T_d)

# Graficar
plt.figure(figsize=(12,8))

plt.subplot(2,1,1)
plt.plot(omega, mag_a, label='Sistema a')
plt.plot(omega, mag_b, label='Sistema b')
plt.plot(omega, mag_c, label='Sistema c')
plt.plot(omega, mag_d, label='Sistema d')
plt.title('Módulo |T(e^{jω})|')
plt.xlabel('ω [rad]')
plt.ylabel('Magnitud')
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(omega, phase_a, label='Sistema a')
plt.plot(omega, phase_b, label='Sistema b')
plt.plot(omega, phase_c, label='Sistema c')
plt.plot(omega, phase_d, label='Sistema d')
plt.title('Fase ∠T(e^{jω})')
plt.xlabel('ω [rad]')
plt.ylabel('Fase [rad]')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()