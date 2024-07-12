import os #Biblioteca para guardar imagenes, gif
import numpy as np #Biblioteca para usar arreglos, matrices, etc.
import matplotlib.pyplot as plt #Biblioteca para generar gráficas
import imageio #Biblioteca para generar gif
import tkinter as tk #Biblioteca para crear interfaces gráficas
from tkinter import ttk, messagebox #Para poder suministrar datos y generar mensajes en ventana
from numba import njit, prange #Biblioteca para poder paralelizar los procesos y mejorar el tiempo de ejecución
#njit hace que numba compile a código maquina, prange se usa para paralelizar los ciclos for

# Función para condiciones iniciales 1, usa parallel=True para paralelizar el proceso usando njit de Numba
@njit(parallel=True)
def condini1(x, y, cmax):
    Z = np.zeros_like(x)
    # Condiciones de frontera, los bordes izquierdo, superior e inferior tendrán el valor de cmax
    # mientas que el borde derecho tendrá el valor de 0
    Z[:, 0] = Z[0, :] = Z[-1, :] = cmax
    Z[:, -1] = 0
    return Z

# Función para condiciones iniciales 1, usa parallel=True para paralelizar el proceso usando njit de Numba
@njit(parallel=True)
def condini2(x, y, cmax):
    Z = np.zeros_like(x)
    # Crear una isla de calor en el centro de la grilla, usando diferentes radios para que no sea un bloque de calor
    # sino una distribución donde el valor máximo cmax se encuentra en el centro y a partir de ahí disminuye el calor
    centro_x = int(x.shape[0] / 2)
    centro_y = int(y.shape[1] / 2)
    radio1 = int(min(x.shape[0], y.shape[1]) / 12)
    radio2 = int(min(x.shape[0], y.shape[1]) / 10)
    radio3 = int(min(x.shape[0], y.shape[1]) / 8)
    radio4 = int(min(x.shape[0], y.shape[1]) / 6)
    # ciclo para asignar distintos calores usando if, y desde afuera hacia adentro
    for i in prange(x.shape[0]):
        for j in prange(y.shape[1]):
            if (i - centro_x) ** 2 + (j - centro_y) ** 2 <= radio4 ** 2:
                Z[i, j] = cmax/6
            if (i - centro_x) ** 2 + (j - centro_y) ** 2 <= radio3 ** 2:
                Z[i, j] = cmax/4
            if (i - centro_x) ** 2 + (j - centro_y) ** 2 <= radio2 ** 2:
                Z[i, j] = cmax/2
            if (i - centro_x) ** 2 + (j - centro_y) ** 2 <= radio1 ** 2:
                Z[i, j] = cmax
    return Z

# Función para condiciones iniciales 1, usa parallel=True para paralelizar el proceso usando njit de Numba
@njit(parallel=True)
def condini3(x, y, cmax):
    Z = np.zeros_like(x)
    # Condiciones de frontera, los bordes izquierdo, superior e inferior tendrán el valor de 0
    # mientas que el borde derecho tendrá el valor de cmax, básicamente lo contrario a condini1
    Z[:, 0] = Z[0, :] = Z[-1, :] = 0
    Z[:, -1] = cmax
    return Z

# Función principal que realiza el cálculo del calor usando el método de Diferencias Finitas centradas
@njit(parallel=True)
def act_calor(Z, k, dt, dx, dy):
    m, n = Z.shape
    Z_nuevo = np.copy(Z)
    # Calculamos el alpha fuera del ciclo for para no hacer cálculos innecesarios
    # puesto que a y b pueden ser distintos necesitamos calcular el alpha para cada eje
    alpha_x = k * dt / dx ** 2
    alpha_y = k * dt / dy ** 2
    # Actualizamos todos los puntos excepto los bordes para mantener las condiciones de frontera
    # se usa prange para paralelizar el proceso y mejorar la eficiencia
    for i in prange(1, m - 1):
        for j in prange(1, n - 1):
            Z_nuevo[i, j] = Z[i, j] + alpha_x * (Z[i + 1, j] + Z[i - 1, j] - 2 * Z[i, j]) \
                                        + alpha_y * (Z[i, j + 1] + Z[i, j - 1] - 2 * Z[i, j])
    return Z_nuevo


# Función para generar las gráficas
def generarimagen(x, y, Z, t, vmax):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150) # Usamos figsize y dpi para que la gráfica se vea bien
    # mantenemos el calor inicial vmax para que las gráficas mantengan concordancia
    c = ax.contourf(x, y, Z, cmap='viridis', levels=20, vmin=0, vmax=vmax)
    fig.colorbar(c)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Onda de calor en t={:.2f}s'.format(t))
    fig.canvas.draw() # Grafica la figura usando canvas
    imagen = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
        fig.canvas.get_width_height()[::-1] + (3,))
    ax.clear()  # Limpiamos el gráfico
    plt.close(fig) # Cerramos la figura para evitar tener datos innecesarios en memoria
    return imagen


# Función que hace los cálculos
def generardatos(x, y, Z, k, dt, t_inicial, t_final, num_pasos, vmax, dx, dy):
    ttotal = t_final - t_inicial
    # calculamos las iteraciones como la razón entre el tiempo total y el paso del tiempo
    iteraciones = int(ttotal / dt)
    # j es la cantidad de datos que se van a generar hasta guardar una imagen para tener 50 imágenes al final
    j = int(iteraciones / num_pasos)
    imagenes = []
    for l in range(iteraciones):
        t = t_inicial + dt * l
        Z = act_calor(Z, k, dt, dx, dy)
        if (l % j) == 0: #si j es divisor de l se guarda una imagen
            imagen = generarimagen(x, y, Z, t, vmax)
            imagenes.append(imagen)
            # Actualizamos la barrad de progreso
            progreso = (l+j) * 100 // iteraciones  # Calculamos el porcentaje de progreso
            progreso_var.set(progreso)
            vent.update_idletasks()
            # Mostramos el progreso en la terminal
            print(f'Progreso: {progreso}%')
    return imagenes

# Función para generar gif, se activa una vez uno clickea un botón
def generargif(condini_func):
    # Tomamos los datos suministrados
    a, b, k = float(entradas[0].get()), float(entradas[1].get()), float(entradas[2].get())
    t_inicial, t_final = float(entradas[3].get()), float(entradas[4].get())
    # La malla, el paso del tiempo y el num de gráficas siempre será igual
    m, dt, num_pasos = 100, 1e-4, 50
    # Se calcula el dx y dy como la razó entre el tamaño de cada lado entre la cantidad de puntos de la grilla
    dx, dy = a / m, b / m
    # El Calor máximo lo fijamos en 500K
    cmax = 500
    # Para poder usar las distintas funciones de condiciones iniciales, guardamos el gif diciendo cuál se uso
    if condini_func == condini1:
        cond = 1
    elif condini_func == condini2:
        cond = 2
    else:
        cond = 3
    # Generamos la malla y sus condiciones iniciales
    x = np.linspace(0, a, m)
    y = np.linspace(0, b, m)
    X, Y = np.meshgrid(x, y)
    Z = condini_func(X, Y, cmax)
    # Generamos las imágenes
    imagenes = generardatos(x, y, Z, k, dt, t_inicial, t_final, num_pasos, cmax, dx, dy)
    # Generar el GIF a partir de las imágenes generadas
    try:
        # Guardar el GIF en la carpeta 'gifs'
        output_dir = os.path.join(os.getcwd(), 'gifs')
        os.makedirs(output_dir, exist_ok=True)
        # En el gif guardado le ponemos los datos suministrados por el usuario, la condición inicial elejida y el método que se uso, en este caso Diferencias Finitas
        gif_name = f'Difusion Termica para una placa de dimensiones {a, b},' \
                   f' constante de difusión k={k} desde t={t_inicial:.2f}s a t={t_final:.2f}s, con la condición inicial {cond}, usando Diferencias Finitas.gif'
        imageio.mimsave(os.path.join(output_dir, gif_name), imagenes, fps=4, loop=0)
    # Manejamos un mensaje en caso de exito y otro en caso de error
        messagebox.showinfo("Éxito", f"GIF generado y guardado en: {os.path.join(output_dir, gif_name)}")
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

# Función para inicializar y ejecutar la interfaz gráfica
vent = tk.Tk()
vent.title("Onda Calor usando Diferencias Finitas")
mainframe = ttk.Frame(vent, padding="10 10 10 10")
mainframe.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
# Nombres de las entradas
etiquetas = ["a (Largo de la placa):", "b (Ancho de la placa):", "k (Constante de difusión):",
             "Tiempo inicial:", "Tiempo final:"]
entradas = []
for i, etiqueta in enumerate(etiquetas):
    ttk.Label(mainframe, text=etiqueta).grid(row=i, column=0, sticky=tk.W)
    entrada = ttk.Entry(mainframe)
    entrada.grid(row=i, column=1)
    entradas.append(entrada)
# Generamos los botones para que el usuario decida que condición inicial usar
ttk.Button(mainframe, text="Generar GIF 1", command=lambda: generargif(condini1)).grid(row=len(etiquetas) , column=0)
ttk.Button(mainframe, text="Generar GIF 2", command=lambda: generargif(condini2)).grid(row=len(etiquetas) , column=1)
ttk.Button(mainframe, text="Generar GIF 3", command=lambda: generargif(condini3)).grid(row=len(etiquetas) , column=2)
# Generamos una barra de progreso
progreso_var = tk.DoubleVar()
ttk.Label(mainframe, text="Progreso:").grid(row=len(etiquetas) + 1, column=0, sticky=tk.W)
progreso_barra = ttk.Progressbar(mainframe, variable=progreso_var, length=200, mode='determinate')
progreso_barra.grid(row=len(etiquetas) + 1, column=1)
# mainloop mantiene la interfaz gráfica activa para poder usarse cuantas veces quiera sin necesidad de reiniciar el código
vent.mainloop()
