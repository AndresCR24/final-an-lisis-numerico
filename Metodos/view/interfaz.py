import tkinter as tk
from tkinter import ttk
from sympy import symbols, sympify, exp, N
import sympy as sp
from Metodos.modelo.Ceros import *
def open_ceros():
    # Cerrar la ventana principal
    root.withdraw()

    # Crear una nueva ventana para los métodos de ceros
    ceros_window = tk.Toplevel()
    ceros_window.title("Métodos de Ceros")
    ceros_window.geometry(root.geometry())

    # Crear botones para los métodos cerrado y abierto
    btn_metodo_cerrado = ttk.Button(ceros_window, text="Método Cerrado", command=lambda: metodo_cerrado(ceros_window))
    btn_metodo_cerrado.pack(expand=True)

    btn_metodo_abierto = ttk.Button(ceros_window, text="Método Abierto", command=metodo_abierto)
    btn_metodo_abierto.pack(expand=True)

    # Botón para regresar al menú principal
    btn_regresar = ttk.Button(ceros_window, text="Regresar al Menú Principal",
                              command=lambda: regresar_al_menu(ceros_window))
    btn_regresar.pack(expand=True)


def regresar_al_menu(window):
    # Cerrar la ventana actual y mostrar la ventana principal
    window.destroy()
    root.deiconify()

def metodo_cerrado(parent_window):
    # Cerrar la ventana actual
    parent_window.withdraw()

    # Crear una nueva ventana para el método cerrado
    cerrado_window = tk.Toplevel()
    cerrado_window.title("Método Cerrado")
    cerrado_window.geometry(parent_window.geometry())

    # Crear botones para Falsa Posición y Bisección
    btn_falsa_posicion = ttk.Button(cerrado_window, text="Falsa Posición", command=open_falsa_posicion)
    btn_falsa_posicion.pack(expand=True)

    btn_biseccion = ttk.Button(cerrado_window, text="Bisección", command=biseccion)
    btn_biseccion.pack(expand=True)

    # Botón para regresar a la ventana de ceros
    btn_regresar = ttk.Button(cerrado_window, text="Regresar a Métodos de Ceros", command=lambda: regresar_a_ceros(cerrado_window, parent_window))
    btn_regresar.pack(expand=True)
def falsa_posicion():
    #Posicion_falsa()
    print("Falsa Posición - Función aún no implementada")
def open_falsa_posicion():
    x, g, m, c, t, v = symbols('x g m c t v')
    falsa_posicion_window = tk.Toplevel()
    falsa_posicion_window.title("Falsa Posición")
    falsa_posicion_window.geometry("300x200")

    # Entrada para la función
    tk.Label(falsa_posicion_window, text="Función:").pack()
    funcion_entry = tk.Entry(falsa_posicion_window)
    funcion_entry.pack()

    # Entrada para el intervalo
    tk.Label(falsa_posicion_window, text="Intervalo Inicial:").pack()
    intervalo_inicial_entry = tk.Entry(falsa_posicion_window)
    intervalo_inicial_entry.pack()

    tk.Label(falsa_posicion_window, text="Intervalo Final:").pack()
    intervalo_final_entry = tk.Entry(falsa_posicion_window)
    intervalo_final_entry.pack()

    # Entrada para la tolerancia
    tk.Label(falsa_posicion_window, text="Tolerancia:").pack()
    tolerancia_entry = tk.Entry(falsa_posicion_window)
    tolerancia_entry.pack()

    # Botón para ejecutar el método
    ttk.Button(falsa_posicion_window, text="Ejecutar Método", command=lambda: ejecutar_falsa_posicion(
        funcion_entry.get(),
        float(intervalo_inicial_entry.get()),
        float(intervalo_final_entry.get()),
        float(tolerancia_entry.get()))).pack()

def ejecutar_falsa_posicion(funcion_str, intervalo_inicial, intervalo_final, tolerancia):
    # Define las variables simbólicas
    g, m, c, t, v = symbols('g m c t v')

    # Reemplaza 'sp.exp' con 'exp' de sympy en la cadena de la función
    funcion_str = funcion_str.replace("sp.exp", "exp")

    # Convierte la cadena de la función en una expresión sympy
    funcion = sympify(funcion_str)

    # Define una función que evalúa esta expresión y la convierte en un valor numérico
    def f(valor):
        # Asegúrate de que la expresión sea evaluada numéricamente
        return funcion.subs(t, valor).evalf()

    # Llama al método de falsa posición
    resultado = Posicion_falsa(f, intervalo_inicial, intervalo_final, tolerancia)
    print("Resultado:", resultado)


def biseccion():
   # Biseccion()
    print("Bisección - Función aún no implementada")
# Define la función regresar_a_ceros
def regresar_a_ceros(child_window, parent_window):
    # Cerrar la ventana actual (método cerrado) y mostrar la ventana de ceros
    child_window.destroy()
    parent_window.deiconify()

def metodo_abierto():
    # Aquí puedes agregar la lógica para el método abierto
    print("Método Abierto")
def open_interpolacion():
    # Aquí puedes agregar la lógica para abrir la interfaz de métodos de interpolación
    print("Interpolación")

def open_ecuaciones():
    # Aquí puedes agregar la lógica para abrir la interfaz de ecuaciones
    print("Ecuaciones")

def open_integracion():
    # Aquí puedes agregar la lógica para abrir la interfaz de métodos de integración
    print("Integración")

# Crear la ventana principal
root = tk.Tk()
root.title("Métodos Numéricos")

# Configurar el tamaño de la ventana
root.geometry("300x200")

# Crear los botones
btn_ceros = ttk.Button(root, text="Ceros", command=open_ceros)
btn_ceros.pack(expand=True)

btn_interpolacion = ttk.Button(root, text="Interpolación", command=open_interpolacion)
btn_interpolacion.pack(expand=True)

btn_ecuaciones = ttk.Button(root, text="Ecuaciones", command=open_ecuaciones)
btn_ecuaciones.pack(expand=True)

btn_integracion = ttk.Button(root, text="Integración", command=open_integracion)
btn_integracion.pack(expand=True)

# Iniciar el bucle principal de Tkinter
root.mainloop()
