import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit, prange
from time import time

# Pré-calculando pontos de quadratura de Gauss-Legendre

class Tranducers:
    def __init__(self, num_arrays, num_layers, transducer_diam, array_radius, array_distance=None, array_angle=0, spacing=1, freqs=40000, phases=0): # Dimensoes em milimetros, ângulos em graus
        self.num_arrays = num_arrays
        self.num_layers = num_layers
        self.transducer_diam = transducer_diam
        self.array_radius = array_radius
        self.center = np.array([0, 0])
        self.spacing = spacing
        self.array_distance = np.array([array_radius for i in range(num_arrays)])
        if type(array_distance) in (int, float):
            self.array_distance = np.array([array_distance for i in range(num_arrays)])
        if type(array_distance) == list:
            self.array_distance = np.array(array_distance)
        self.array_angle = np.array([2 * np.pi * array / num_arrays + np.pi for array in range(num_arrays)])
        if type(array_angle) == list:
            self.array_angle = 2 * np.pi / 360 * np.array(array_angle)
        self.freqs = np.array([freqs for i in range(num_arrays)])
        if type(freqs) == list:
            self.freqs = np.array(freqs)
        self.phases = np.array([phases for i in range(num_arrays)])
        if type(phases) == list:
            self.phases = np.array(phases)

    def mount(self):
        num_arrays = self.num_arrays
        num_layers = self.num_layers
        t_diam = self.transducer_diam
        array_radius = self.array_radius
        center = self.center
        spacing = self.spacing
        array_distance = self.array_distance
        array_angle = self.array_angle
        freqs = self.freqs
        phases = self.phases
        return self._mount_njit(num_arrays, num_layers, t_diam, array_radius, center, spacing, array_distance, array_angle, freqs,
                   phases)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _mount_njit(num_arrays, num_layers, t_diam, array_radius, center, spacing, array_distance, array_angle, freqs, phases):
        try:
            delta_alpha = np.acos(1 - 0.5 * (((t_diam + spacing) / array_radius) ** 2))
            delta_beta = -np.acos(1 - 0.5 * (((t_diam + spacing) * 3 ** 0.5 / 2) / array_radius) ** 2)
        except:
            delta_alpha, delta_beta = (0, 0)
            pass

        num_transducers = int(num_arrays * (6 * (num_layers / 2 * (num_layers - 1)) + 1))
        transducers = np.zeros((num_transducers, 7))

        count = 0
        for array in prange(num_arrays):
            for row in prange(num_layers):
                t_per_layer = num_layers * 2 - (row + 1)
                for t in prange(t_per_layer):
                    alpha = delta_alpha * (t - (t_per_layer - 1) / 2) + array_angle[array]
                    beta = delta_beta * row

                    transducers[count, 0] = center[0] + array_radius * np.sin(alpha) * np.cos(beta) + (
                                array_distance[array] - array_radius) * np.sin(array_angle[array]) + \
                                            max(array_distance)
                    transducers[count, 1] = center[1] - array_radius * np.cos(alpha) * np.cos(beta) - (
                                array_distance[array] - array_radius) * np.cos(array_angle[array]) + \
                                            max(array_distance)
                    transducers[count, 2] = - array_radius * np.sin(beta)
                    transducers[count, 3] = alpha
                    transducers[count, 4] = beta
                    transducers[count, 5] = freqs[array]
                    transducers[count, 6] = phases[array]
                    count += 1

                    if beta != 0:
                        transducers[count, 0] = center[0] + array_radius * np.sin(alpha) * np.cos(beta) + (
                                array_distance[array] - array_radius) * np.sin(array_angle[array]) + \
                                            max(array_distance)
                        transducers[count, 1] = center[1] - array_radius * np.cos(alpha) * np.cos(beta) - (
                                array_distance[array] - array_radius) * np.cos(array_angle[array]) + \
                                            max(array_distance)
                        transducers[count, 2] = array_radius * np.sin(beta)
                        transducers[count, 3] = alpha
                        transducers[count, 4] = - beta
                        transducers[count, 5] = freqs[array]
                        transducers[count, 6] = phases[array]
                        count += 1

        return transducers

class AcousticField:
    def __init__(self, transducers, rho0=1.2, c0=340, v0=1):
        self.transducers = transducers
        start_time = time()
        self.transducers_mounted = transducers.mount()
        end_time = time()
        print(f"Tempo de execução da montagem: {end_time - start_time:.2f} segundos")
        self.view_size = 2 * max(transducers.array_distance) # mm
        self.resolution = 0.5 # mm
        self.mtx_size = int(self.view_size // self.resolution)
        self.rho0 = rho0
        self.c0 = c0
        self.v0 = v0

    def calculate_pressure(self, num_points=6, time_=0):
        transducers = self.transducers_mounted
        t_diam = self.transducers.transducer_diam
        resolution = self.resolution
        self.mtx_size = int(self.view_size // resolution)
        mtx_size = self.mtx_size
        self.view_size_corrected = mtx_size * resolution
        rho0 = self.rho0
        c0 = self.c0
        gl_points, gl_weights = np.polynomial.legendre.leggauss(num_points)
        self.pressure_field = self._calc_pressure(transducers, t_diam, resolution, mtx_size, time_, rho0, c0, gl_points, gl_weights)

    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _calc_pressure(transducers, t_diam, resolution, mtx_size, time, rho0, c0, gl_points, gl_weights):
        radius = t_diam / 2000 # Convertendo para metros
        pressure_field = np.zeros((mtx_size, mtx_size), dtype=np.complex64)
        for i in prange(mtx_size):
            x = i * resolution / 1000  # Convertendo para metros
            for j in prange(mtx_size):
                y = j * resolution / 1000  # Convertendo para metros
                z = 0  # Plano z fixo em 0 mm
                pd = 0 + 0j
                for t in transducers:
                    tx, ty, tz, alpha, beta, freq, phase = t
                    omega = freq * 2.0 * np.pi  # rad/s
                    wave_num = omega / c0  # rad/m
                    wave_len = c0 / freq
                    tx /= 1000  # Convertendo para metros
                    ty /= 1000
                    tz /= 1000
                    r_vec_x = x - tx
                    r_vec_y = y - ty
                    r_vec_z = z - tz
                    r = np.sqrt(r_vec_x ** 2 + r_vec_y ** 2 + r_vec_z ** 2) + 1e-10
                    area_vec = np.array([
                        -np.sin(alpha) * np.cos(beta),
                        np.cos(alpha) * np.cos(beta),
                        np.sin(beta)
                    ])
                    sin_phi = np.sqrt(1 - ((r_vec_x * area_vec[0] + r_vec_y * area_vec[1] + r_vec_z * area_vec[2]) / r) ** 2) # Calculando o seno de phi através do produto escalar

                    integral = 0 + 0j
                    for m in prange(len(gl_points)):
                        for n in prange(len(gl_points)):
                            rho = gl_points[n] * radius/2 + radius/2
                            theta = gl_points[m] * np.pi + np.pi
                            weight = gl_weights[n] * gl_weights[m]
                            r0 = np.sqrt(r ** 2 + rho ** 2 - 2 * r * rho * np.cos(theta) * sin_phi)
                            integrand = np.exp(-1j * wave_num * r0) * rho / r0
                            integral += integrand * weight * np.pi * radius/2

                    pd += 1j * 96500 * np.exp(1j * (omega * time + phase)) * integral

                pressure_field[i, j] = pd
        return pressure_field

    def calculate_velocity(self):
        dx = dy = self.resolution / 1000
        gradient_x, gradient_y = np.gradient(np.abs(self.pressure_field)/0.707, dx, dy)
        omega = 2 * np.pi * np.mean(self.transducers.freqs)
        velocity_x = -1j / (omega * self.rho0) * gradient_x
        velocity_y = -1j / (omega * self.rho0) * gradient_y
        self.velocity_field = (velocity_x, velocity_y)

    def calculate_potential(self, object_radius=1e-3): # Raio do objeto em metros
        potential = 2 * np.pi * object_radius ** 3 * 0.5 * (np.abs(self.pressure_field) ** 2 / (3 * self.rho0 * self.c0 ** 2) - 0.5 * self.rho0 * np.abs(self.velocity_field[0]**2 + self.velocity_field[1]**2))
        self.potential_field = potential

    def calculate_force(self):
        dx = dy = self.resolution / 1000
        gradient_x, gradient_y = np.gradient(self.potential_field, dx, dy)
        force = - gradient_y
        self.force = force

    def calculate_all(self, num_points=6):
        self.calculate_pressure(num_points)
        self.calculate_velocity()
        self.calculate_potential()
        self.calculate_force()

    def plot_field(self, field_name, num_points=6):
        start_time = time()
        self.calculate_all(num_points)
        end_time = time()
        fig, ax = plt.subplots()
        field = np.abs(self.pressure_field).T
        view_size = self.view_size_corrected
        extent = [-view_size / 2, view_size / 2, -view_size / 2, view_size / 2]
        im = ax.imshow(field, animated=True, extent=extent, origin='lower', cmap='turbo')
        cbar = fig.colorbar(im, ax=ax, ticks=np.linspace(field.min(), field.max(), num=5))
        cbar.ax.set_title('Pressão Eficaz (Pa)', pad=15, loc='left', fontsize='medium')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Campo de Acústico', pad=20)
        print(f"Tempo de execução do cálculo: {end_time - start_time:.2f} segundos")
        start_time = time()
        plt.show()
        end_time = time()
        print(f"Tempo para plotar: {end_time - start_time:.2f} segundos")

    def animation_freq_change(self, change_rate, period=100): # Taxa de mudança em Hz por segundo, período em segundos
        if type(change_rate) in (int, float):
            change_rate = np.array([change_rate for i in range(self.transducers.num_arrays)])
        self.change_rate = np.array(change_rate)
        self.calculate_pressure()
        fig, ax = plt.subplots()
        field = np.abs(self.pressure_field).T
        view_size = self.view_size_corrected
        extent = [-view_size / 2, view_size / 2, -view_size / 2, view_size / 2]
        self.freq_im = ax.imshow(field, animated=True, extent=extent, origin='lower', cmap='turbo')
        fig.colorbar(self.freq_im, ax=ax)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        self.time_ = 0
        self.freq_ani = animation.FuncAnimation(fig, self.update, frames=period*5, blit=True, interval=500)  # intervalo ajustado para 100ms
        self.freq_ani.save(r'E:\Documentos\0 - Levitador Acustico\Imagens\GIFs\class_test2.gif') # Alterar caminho
        plt.show()

    def update(self, period):
        freq_change_rate = self.change_rate
        start_time = time()
        transducers = self.transducers
        transducers.freqs = transducers.freqs + freq_change_rate * self.time_
        self.transducers_mounted = transducers.mount()
        self.calculate_pressure(time_=self.time_)
        self.time_ += 0.5
        end_time = time()
        print(f"Tempo de execução: {end_time - start_time:.2f} segundos")
        self.freq_im.set_array(np.abs(self.pressure_field).T)
        return [self.freq_im]

# Exemplo de uso do programa
transd = Tranducers(num_arrays=3,num_layers=4,transducer_diam=10,array_radius=85,array_distance=67)
# transd = Tranducers(1,1,10,40,300)
fields = AcousticField(transd)
fields.resolution = 0.5 # mm
fields.plot_field('pressure')
