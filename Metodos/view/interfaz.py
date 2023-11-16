import tkinter as tk
from tkinter import messagebox
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify, exp
from Metodos.modelo.Ceros import *

def open_ceros_window():
    # Ocultar la ventana principal
    root.withdraw()

    # Crear una nueva ventana para métodos de ceros
    ceros_window = tk.Toplevel()
    ceros_window.title("Métodos de Ceros")

    # Agregar botones para Método cerrado y Método abierto
    metodo_cerrado_button = tk.Button(ceros_window, text="Método cerrado", command=lambda: metodo_selected(ceros_window, "cerrado"))
    metodo_cerrado_button.pack()

    metodo_abierto_button = tk.Button(ceros_window, text="Método abierto", command=lambda: metodo_selected(ceros_window, "abierto"))
    metodo_abierto_button.pack()

    # Agregar botón para regresar
    regresar_button = tk.Button(ceros_window, text="Regresar", command=lambda: regresar(root, ceros_window))
    regresar_button.pack()

def metodo_selected(previous_window, metodo):
    # Cerrar la ventana anterior
    previous_window.destroy()

    if metodo == "cerrado":
        # Crear una nueva ventana para métodos cerrados
        cerrado_window = tk.Toplevel()
        cerrado_window.title("Métodos Cerrados")

        # Modificar aquí el botón para Falsa Posición
        falsa_pos_button = tk.Button(cerrado_window, text="Falsa Posición", command=abrir_falsa_posicion)
        falsa_pos_button.pack()

        biseccion_button = tk.Button(cerrado_window, text="Bisección", command=lambda: print("Bisección seleccionada"))
        biseccion_button.pack()

        # Agregar botón para regresar
        regresar_button = tk.Button(cerrado_window, text="Regresar", command=lambda: regresar(root, cerrado_window))
        regresar_button.pack()

    elif metodo == "abierto":
        # Implementar lógica para método abierto
        print("Método abierto seleccionado")

def regresar(main_window, current_window):
    # Cerrar la ventana actual y mostrar la ventana principal
    current_window.destroy()
    main_window.deiconify()


def ejecutar_falsa_posicion(funcion_str, a_val, b_val, tol_val):
    x = sp.symbols('x')
    funcion_str = funcion_str.replace('sp.', '')

    try:
        funcion = lambdify(x, parse_expr(funcion_str))
        resultado = Posicion_falsa(funcion, a_val, b_val, tol_val)
        return resultado
    except Exception as e:
        return str(e)


def abrir_falsa_posicion():
    def comando_ejecutar():
        funcion_str = funcion_entry.get()
        a_val = float(intervalo_a_entry.get())
        b_val = float(intervalo_b_entry.get())
        tol_val = float(tol_entry.get())

        resultado = ejecutar_falsa_posicion(funcion_str, a_val, b_val, tol_val)
        resultado_label.config(text=f"Resultado: {resultado}")

    falsa_pos_window = tk.Toplevel()
    falsa_pos_window.title("Falsa Posición")

    tk.Label(falsa_pos_window, text="Función:").pack()
    funcion_entry = tk.Entry(falsa_pos_window, width=50)
    funcion_entry.pack()

    tk.Label(falsa_pos_window, text="Intervalo a:").pack()
    intervalo_a_entry = tk.Entry(falsa_pos_window, width=50)
    intervalo_a_entry.pack()

    tk.Label(falsa_pos_window, text="Intervalo b:").pack()
    intervalo_b_entry = tk.Entry(falsa_pos_window, width=50)
    intervalo_b_entry.pack()

    tk.Label(falsa_pos_window, text="Tolerancia:").pack()
    tol_entry = tk.Entry(falsa_pos_window, width=50)
    tol_entry.pack()

    ejecutar_button = tk.Button(falsa_pos_window, text="Ejecutar", command=comando_ejecutar)
    ejecutar_button.pack()

    resultado_label = tk.Label(falsa_pos_window, text="Resultado:")
    resultado_label.pack()


#----------------------------
# Crear la ventana principal
root = tk.Tk()
root.title("Métodos Numéricos")

# Crear un botón y añadirlo a la ventana principal
ceros_button = tk.Button(root, text="Ceros", command=open_ceros_window)
ceros_button.pack()
# Ejecutar el bucle principal de la ventana
root.geometry("300x200")
root.mainloop()
