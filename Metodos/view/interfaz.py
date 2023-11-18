import tkinter as tk
from tkinter import messagebox
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify, exp, symbols, Matrix, sympify

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import pandas as pd
from tkinter import Toplevel
from Metodos.modelo.Ceros import *
from Metodos.modelo.ecuaciones import *

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

    metodo_grafico_button = tk.Button(ceros_window, text="Grafica", command=lambda: metodo_selected(ceros_window, "grafica"))
    metodo_grafico_button.pack()
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

        biseccion_button = tk.Button(cerrado_window, text="Bisección", command=abrir_biseccion)
        biseccion_button.pack()

        # Agregar botón para regresar
        regresar_button = tk.Button(cerrado_window, text="Regresar", command=lambda: regresar(root, cerrado_window))
        regresar_button.pack()

    elif metodo == "abierto":
        abierto_window = tk.Toplevel()
        abierto_window.title("Metodos Abiertos")

        newton_button = tk.Button(abierto_window, text="Newton", command= abrir_newton)
        newton_button.pack()

        secante_button = tk.Button(abierto_window, text="Secante", command= abrir_secante)
        secante_button.pack()

        regresar_button = tk.Button(abierto_window, text="Regresar", command=lambda: regresar(root, abierto_window))
        regresar_button.pack()
    elif metodo == "grafica":
        grafica_window = tk.Toplevel()
        grafica_window.title("Grafica")

        grafica_button = tk.Button(grafica_window, text="Realizar grafica", command= abrir_graficador)
        grafica_button.pack()


def regresar(main_window, current_window):
    # Cerrar la ventana actual y mostrar la ventana principal
    current_window.destroy()
    main_window.deiconify()

#Metodo Cerrado Falsa Posición
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
#Metodo cerrado Biseccion
def ejecutar_biseccion(funcion_str, a_val, b_val, tol_val):
    x = sp.symbols('x')
    funcion_str = funcion_str.replace('sp.', '')

    try:
        funcion = lambdify(x, parse_expr(funcion_str))
        resultado = Biseccion(funcion, a_val, b_val, tol_val)
        return resultado
    except Exception as e:
        return str(e)


def abrir_biseccion():
    def comando_ejecutar():
        funcion_str = funcion_entry.get()
        a_val = float(intervalo_a_entry.get())
        b_val = float(intervalo_b_entry.get())
        tol_val = float(tol_entry.get())

        resultado = ejecutar_biseccion(funcion_str, a_val, b_val, tol_val)
        resultado_label.config(text=f"Resultado: {resultado}")

    biseccion_window = tk.Toplevel()
    biseccion_window.title("Bisección")

    tk.Label(biseccion_window, text="Función:").pack()
    funcion_entry = tk.Entry(biseccion_window, width=50)
    funcion_entry.pack()

    tk.Label(biseccion_window, text="Intervalo a:").pack()
    intervalo_a_entry = tk.Entry(biseccion_window, width=50)
    intervalo_a_entry.pack()

    tk.Label(biseccion_window, text="Intervalo b:").pack()
    intervalo_b_entry = tk.Entry(biseccion_window, width=50)
    intervalo_b_entry.pack()

    tk.Label(biseccion_window, text="Tolerancia:").pack()
    tol_entry = tk.Entry(biseccion_window, width=50)
    tol_entry.pack()

    ejecutar_button = tk.Button(biseccion_window, text="Ejecutar", command=comando_ejecutar)
    ejecutar_button.pack()

    resultado_label = tk.Label(biseccion_window, text="Resultado:")
    resultado_label.pack()

def ejecutar_newton(funcion_str, x0, tol_val):
    # Convertir la cadena de la función en una función simbólica
    funcion = sp.sympify(funcion_str)

    # Convertir x0 y tol_val a tipos de datos adecuados si es necesario
    x0 = float(x0)
    tol_val = float(tol_val)

    # Llamar al método de Newton de Ceros.py
    resultado = Newton(funcion, x0, tol_val)

    # Devolver el resultado
    return resultado
def abrir_newton():
    def comando_ejecutar():
        funcion_str = funcion_entry.get()
        x0 = float(x0_entry.get())
        tol_val = float(tol_entry.get())

        resultado = ejecutar_newton(funcion_str, x0, tol_val)
        resultado_label.config(text=f"Resultado: {resultado}")

    newton_window = tk.Toplevel()
    newton_window.title("Método de Newton")

    tk.Label(newton_window, text="Función (Colocarla normal directamente sin el sp.):").pack()
    funcion_entry = tk.Entry(newton_window, width=50)
    funcion_entry.pack()

    tk.Label(newton_window, text="Valor inicial (x0):").pack()
    x0_entry = tk.Entry(newton_window, width=50)
    x0_entry.pack()

    tk.Label(newton_window, text="Tolerancia:").pack()
    tol_entry = tk.Entry(newton_window, width=50)
    tol_entry.pack()

    ejecutar_button = tk.Button(newton_window, text="Ejecutar", command=comando_ejecutar)
    ejecutar_button.pack()

    resultado_label = tk.Label(newton_window, text="Resultado:")
    resultado_label.pack()

#Metodo de la secante
def ejecutar_secante(funcion_str, x0, x1, tol_val):
    x = sp.symbols('x')
    funcion_simbolica = sp.sympify(funcion_str)
    funcion_numerica = sp.lambdify(x, funcion_simbolica, 'numpy')  # Convierte a función numérica

    resultado = Secante(funcion_numerica, x0, x1, tol_val)
    return resultado

# Función para abrir la ventana del método de la Secante
def abrir_secante():
    def comando_ejecutar():
        funcion_str = funcion_entry.get()
        try:
            x0 = float(x0_entry.get())
            x1 = float(x1_entry.get())
            tol_val = float(tol_entry.get())
        except ValueError:
            print("Error: x0, x1 y tolerancia deben ser números.")
            return

        resultado = ejecutar_secante(funcion_str, x0, x1, tol_val)
        resultado_label.config(text=f"Resultado: {resultado}")
        print('La raíz es:', resultado)

    secante_window = tk.Toplevel()
    secante_window.title("Método de la Secante")

    tk.Label(secante_window, text="Función:").pack()
    funcion_entry = tk.Entry(secante_window, width=50)
    funcion_entry.pack()

    tk.Label(secante_window, text="x0:").pack()
    x0_entry = tk.Entry(secante_window, width=50)
    x0_entry.pack()

    tk.Label(secante_window, text="x1:").pack()
    x1_entry = tk.Entry(secante_window, width=50)
    x1_entry.pack()

    tk.Label(secante_window, text="Tolerancia:").pack()
    tol_entry = tk.Entry(secante_window, width=50)
    tol_entry.pack()

    ejecutar_button = tk.Button(secante_window, text="Ejecutar", command=comando_ejecutar)
    ejecutar_button.pack()

    resultado_label = tk.Label(secante_window, text="Resultado:")
    resultado_label.pack()

#Crear grafica Ceros
def graficar_funcion():
    funcion_str = entrada_funcion.get()
    try:
        # Convierte la cadena de entrada en una función numérica
        valor_final_rango = float(entrada_valor_final.get())

        x = sp.symbols('x')
        funcion_simbolica = parse_expr(funcion_str, evaluate=False)
        funcion_numerica = lambdify(x, funcion_simbolica, modules=["numpy", "math"])

        # Definición del rango de valores para t
        u = np.arange(0, valor_final_rango, 0.01)
        ventana_grafica = tk.Toplevel()
        ventana_grafica.title("Gráfica de la Función")

        # Creando la figura de Matplotlib
        fig, ax = plt.subplots()
        ax.plot(u, funcion_numerica(u))
        ax.axhline(0, color='orange')
        ax.grid()

        # Añadiendo la figura a la ventana de Tkinter
        canvas = FigureCanvasTkAgg(fig, master=ventana_grafica)
        canvas_widget = canvas.get_tk_widget()

        # Configurando el widget para que se redimensione con la ventana
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Mostrar la gráfica
        canvas.draw()
    except Exception as e:
        print("Error al graficar la función:", e)

# Función para abrir la ventana de graficación
def abrir_graficador():
    graficador_window = tk.Toplevel()
    graficador_window.title("Graficador de Funciones")

    tk.Label(graficador_window, text="Ingrese la función f(t):").pack()
    global entrada_funcion, entrada_valor_final
    entrada_funcion = tk.Entry(graficador_window, width=50)
    entrada_funcion.pack()
    #entrada_funcion = tk.Entry(graficador_window)
    #entrada_funcion.pack()

    tk.Label(graficador_window, text="Ingrese el valor final del rango:").pack()
    entrada_valor_final = tk.Entry(graficador_window, width=50)
    entrada_valor_final.pack()
    #entrada_valor_final = tk.Entry(graficador_window)
    #entrada_valor_final.pack()

    boton_graficar = tk.Button(graficador_window, text="Graficar", command=graficar_funcion)
    boton_graficar.pack()

#Interpolacion
def open_ecuaciones_window():
    # Ocultar la ventana principal
    root.withdraw()

    # Crear una nueva ventana para métodos
    ecuaciones_window = tk.Toplevel()
    ecuaciones_window.title("Métodos de Ecuaciones diferenciales")

    # Agregar botones para Método cerrado y Método abierto
    metodo_edo_button = tk.Button(ecuaciones_window, text="Ecuaciones diferenciales de orden 1", command=lambda: metodo_selected_ecuaciones(ecuaciones_window, "orden1"))
    metodo_edo_button.pack()

    metodo_edo2_button = tk.Button(ecuaciones_window, text="Ecuaciones diferenciales de orden 2", command=lambda: metodo_selected_ecuaciones(ecuaciones_window, "orden2"))
    metodo_edo2_button.pack()

   # metodo_grafico_button = tk.Button(ecuaciones_window, text="Grafica", command=lambda: metodo_selected(ecuaciones_window, "grafica"))
    #metodo_grafico_button.pack()
    # Agregar botón para regresar
    regresar_button = tk.Button(ecuaciones_window, text="Regresar", command=lambda: regresar(root, ecuaciones_window))
    regresar_button.pack()

def metodo_selected_ecuaciones(previous_window, metodo):
    # Cerrar la ventana anterior
    previous_window.destroy()

    if metodo == "orden1":
        # Crear una nueva ventana para métodos cerrados
        cerrado_window = tk.Toplevel()
        cerrado_window.title("Ecuaciones diferenciales orden 1")

        # Modificar aquí el botón para Falsa Posición
        euler_pos_button = tk.Button(cerrado_window, text="Euler", command=abrir_euler_ecuacion)
        euler_pos_button.pack()

        runge_kutta_orden1 = tk.Button(cerrado_window, text="Runge Kutta", command=abrir_runge_kutta_ecuacion)
        runge_kutta_orden1.pack()


        # Agregar botón para regresar
        regresar_button = tk.Button(cerrado_window, text="Regresar", command=lambda: regresar(root, cerrado_window))
        regresar_button.pack()

    elif metodo == "orden2":
        abierto_window = tk.Toplevel()
        abierto_window.title("Ecuaciones diferenciales orden 2")

        euler_button_orden2 = tk.Button(abierto_window, text="Euler", command= abrir_euler_2do_orden)
        euler_button_orden2.pack()

        runge_kutta_orden2 = tk.Button(abierto_window, text="Runge Kutta", command= abrir_runge_kutta_2do_orden())
        runge_kutta_orden2.pack()

        regresar_button = tk.Button(abierto_window, text="Regresar", command=lambda: regresar(root, abierto_window))
        regresar_button.pack()

#Euler Ecuaciones diferenciales orden 1
def ejecutar_euler_ecuacion(funcion_str, a_val, b_val, h, co_val):
    t, y = symbols('t y')  # Definimos dos símbolos para t y y
    # Reemplazamos 'np.' por nada ya que SymPy usa 'exp' directamente
    funcion_str = funcion_str.replace('np.', '')  # Si el usuario usa 'np.exp', lo cambiamos solo a 'exp'

    try:
        # Parseamos la cadena de la función asegurándonos de que es una expresión de SymPy válida
        funcion = lambdify((t, y), parse_expr(funcion_str), modules='numpy')  # Convertimos la cadena en una función
        t_values, y_values = Euler(funcion, a_val, b_val, h, co_val)  # Llamamos a la función Euler
        # Crear un DataFrame con los resultados
        data = {'Time': t_values, 'Euler_Aproximacion': y_values}
        resultado_df = pd.DataFrame(data)
        return resultado_df
    except Exception as e:
        return pd.DataFrame(columns=['Error'], data=[[str(e)]])

def abrir_euler_ecuacion():
    def comando_ejecutar():
        funcion_str = funcion_entry.get()
        a_val = float(intervalo_a_entry.get())
        b_val = float(intervalo_b_entry.get())
        h = float(tol_entry.get())
        co_val = float(co_entry.get())

        resultado_df = ejecutar_euler_ecuacion(funcion_str, a_val, b_val, h, co_val)
        mostrar_resultados(resultado_df)
        if resultado_df.empty:
            resultado_label.config(text="Error en la ejecución")
        else:
            resultado_label.config(text="Ejecución exitosa")
            # Guardamos los resultados en una variable accesible para la función de graficar
            abrir_euler_ecuacion.resultado_df = resultado_df

    def graficar_resultados():
        # Accedemos a los resultados almacenados previamente
        df = abrir_euler_ecuacion.resultado_df
        if df is not None and not df.empty:
            # Creamos una figura de matplotlib
            fig = Figure(figsize=(6, 4), dpi=100)
            plot = fig.add_subplot(1, 1, 1)
            # Graficamos los resultados y añadimos un título al gráfico
            plot.plot(df['Time'], df['Euler_Aproximacion'], '-o', color='blue')
            plot.set_title('Aproximación Euler')  # Título del gráfico

            # Creamos una ventana de Tkinter para el gráfico
            grafico_window = tk.Toplevel()
            grafico_window.title("Gráfico de Resultados Euler")
            canvas = FigureCanvasTkAgg(fig, master=grafico_window)
            canvas.draw()

            # Creamos un widget de Canvas de Tkinter con el gráfico de matplotlib y lo empaquetamos para que se expanda y llene
            widget_canvas = canvas.get_tk_widget()
            widget_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Para que el gráfico se reajuste cuando la ventana cambia de tamaño, vinculamos el cambio de tamaño de la ventana con una actualización del canvas
            def on_resize(event):
                canvas.draw()

            grafico_window.bind('<Configure>', on_resize)

        else:
            tk.messagebox.showerror("Error", "No hay datos para graficar.")

    def mostrar_resultados(df):
        resultado_window = tk.Toplevel()
        resultado_window.title("Resultados de Euler")
        text = tk.Text(resultado_window)
        text.pack()
        text.insert(tk.END, df.to_string(index=False))  # Convertir DataFrame a string para mostrarlo

    euler_window = tk.Toplevel()
    euler_window.title("Método de Euler")

    tk.Label(euler_window, text="Función:").pack()
    funcion_entry = tk.Entry(euler_window, width=50)
    funcion_entry.pack()

    tk.Label(euler_window, text="Intervalo a:").pack()
    intervalo_a_entry = tk.Entry(euler_window, width=50)
    intervalo_a_entry.pack()

    tk.Label(euler_window, text="Intervalo b:").pack()
    intervalo_b_entry = tk.Entry(euler_window, width=50)
    intervalo_b_entry.pack()

    tk.Label(euler_window, text="Tamaño de paso:").pack()
    tol_entry = tk.Entry(euler_window, width=50)
    tol_entry.pack()

    tk.Label(euler_window, text="Condición Inicial:").pack()
    co_entry = tk.Entry(euler_window, width=50)
    co_entry.pack()

    ejecutar_button = tk.Button(euler_window, text="Ejecutar", command=comando_ejecutar)
    ejecutar_button.pack()

    resultado_label = tk.Label(euler_window, text="Resultado:")
    resultado_label.pack()

    # Botón para graficar los resultados
    graficar_button = tk.Button(euler_window, text="Graficar", command=graficar_resultados)
    graficar_button.pack()

    #grafico_button = tk.Button(euler_window, text="Grafico", command=comando_grafico_euler)
    #grafico_button.pack()

abrir_euler_ecuacion.resultado_df = None

#Runge kutta Ecuaciones dieferenciales orden 1
def ejecutar_runge_kutta(funcion_str, a_val, b_val, h, co_val):
    t, y = symbols('t y')  # Define dos símbolos para t y y
    funcion = lambdify((t, y), parse_expr(funcion_str), modules='numpy')  # Convierte la cadena en una función

    try:
        t_values, y_values = runge_kutta(funcion, a_val, b_val, h, co_val)  # Llama a la función de Runge-Kutta
        # Crea un DataFrame con los resultados
        data = {'Time': t_values, 'Runge_Kutta_Aproximacion': y_values}
        resultado_df = pd.DataFrame(data)
        return resultado_df
    except Exception as e:
        return pd.DataFrame(columns=['Error'], data=[[str(e)]])

def abrir_runge_kutta_ecuacion():
    def comando_ejecutar():
        funcion_str = funcion_entry.get()
        a_val = float(intervalo_a_entry.get())
        b_val = float(intervalo_b_entry.get())
        h = float(tol_entry.get())
        co_val = float(co_entry.get())

        resultado_df = ejecutar_runge_kutta(funcion_str, a_val, b_val, h, co_val)
        abrir_runge_kutta_ecuacion.resultado_df = resultado_df

        if not resultado_df.empty:
            mostrar_resultados(resultado_df)
        else:
            resultado_label.config(text="Error en la ejecución")

    def mostrar_resultados(df):
        resultado_window = tk.Toplevel()
        resultado_window.title("Resultados de Runge-Kutta")
        text = tk.Text(resultado_window)
        text.pack()
        text.insert(tk.END, df.to_string(index=False))  # Convertir DataFrame a string para mostrarlo



    def graficar_resultados():
        df = abrir_runge_kutta_ecuacion.resultado_df
        if df is not None and not df.empty:
            fig = Figure(figsize=(6, 4), dpi=100)
            plot = fig.add_subplot(1, 1, 1)
            plot.plot(df['Time'], df['Runge_Kutta_Aproximacion'], '-o', color='red')
            plot.set_title('Aproximación Runge-Kutta')

            grafico_window = tk.Toplevel()
            grafico_window.title("Gráfico de Resultados Runge-Kutta")
            canvas = FigureCanvasTkAgg(fig, master=grafico_window)
            canvas.draw()
            widget_canvas = canvas.get_tk_widget()
            widget_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            def on_resize(event):
                canvas.draw()
            grafico_window.bind('<Configure>', on_resize)
        else:
            tk.messagebox.showerror("Error", "No hay datos para graficar.")

    runge_kutta_window = tk.Toplevel()
    runge_kutta_window.title("Método de Runge-Kutta")

    tk.Label(runge_kutta_window, text="Función:").pack()
    funcion_entry = tk.Entry(runge_kutta_window, width=50)
    funcion_entry.pack()

    tk.Label(runge_kutta_window, text="Intervalo a:").pack()
    intervalo_a_entry = tk.Entry(runge_kutta_window, width=50)
    intervalo_a_entry.pack()

    tk.Label(runge_kutta_window, text="Intervalo b:").pack()
    intervalo_b_entry = tk.Entry(runge_kutta_window, width=50)
    intervalo_b_entry.pack()

    tk.Label(runge_kutta_window, text="Tamaño de paso:").pack()
    tol_entry = tk.Entry(runge_kutta_window, width=50)
    tol_entry.pack()

    tk.Label(runge_kutta_window, text="Condición Inicial:").pack()
    co_entry = tk.Entry(runge_kutta_window, width=50)
    co_entry.pack()

    ejecutar_button = tk.Button(runge_kutta_window, text="Ejecutar", command=comando_ejecutar)
    ejecutar_button.pack()

    graficar_button = tk.Button(runge_kutta_window, text="Graficar", command=graficar_resultados)
    graficar_button.pack()

    resultado_label = tk.Label(runge_kutta_window, text="Resultado:")
    resultado_label.pack()

abrir_runge_kutta_ecuacion.resultado_df = None
#Euler para ecuaciones diferenciales de orden 2
def ejecutar_euler_2do_orden(f1_str, f2_str, a_val, b_val, h, co_val):
    t, x, y = symbols('t x y')  # Definimos tres símbolos para t, x, y
    f1_expr = sympify(f1_str.replace('np.', ''))
    f2_expr = sympify(f2_str.replace('np.', ''))

    # Convertimos las expresiones en funciones
    f1_func = lambdify((t, x, y), f1_expr, modules='numpy')
    f2_func = lambdify((t, x, y), f2_expr, modules='numpy')

    def f(t, u):
        return np.array([f1_func(t, u[0], u[1]), f2_func(t, u[0], u[1])])

    try:
        t_values, y_values = Euler(f, a_val, b_val, h, np.array(co_val))
        # Crear un DataFrame con los resultados
        data = {'Time': t_values, 'f1(t)': [y[0] for y in y_values], "f2(t)": [y[1] for y in y_values]}
        resultado_df = pd.DataFrame(data)
        return resultado_df
    except Exception as e:
        return pd.DataFrame(columns=['Error'], data=[[str(e)]])

def abrir_euler_2do_orden():
    def comando_ejecutar():
        f1_str = f1_entry.get()
        f2_str = f2_entry.get()
        a_val = float(intervalo_a_entry.get())
        b_val = float(intervalo_b_entry.get())
        h = float(tol_entry.get())
        co_val_x = float(co_x_entry.get())
        co_val_y = float(co_y_entry.get())

        resultado_df = ejecutar_euler_2do_orden(f1_str, f2_str, a_val, b_val, h, [co_val_x, co_val_y])
        mostrar_resultados(resultado_df)
        if resultado_df.empty:
            resultado_label.config(text="Error en la ejecución")
        else:
            resultado_label.config(text="Ejecución exitosa")
            abrir_euler_2do_orden.resultado_df = resultado_df

    def graficar_resultados():
        df = abrir_euler_2do_orden.resultado_df
        if df is not None and not df.empty:
            fig = Figure(figsize=(6, 4), dpi=100)
            plot = fig.add_subplot(1, 1, 1)
            plot.plot(df['Time'], df['f1(t)'], '-o', color='blue', label='x(t)')
            plot.plot(df['Time'], df['f2(t)'], '-o', color='red', label='y(t)')
            plot.set_title('Aproximación Euler')
            plot.set_xlabel('Time')
            plot.set_ylabel('Values')
            plot.legend()

            grafico_window = tk.Toplevel()
            grafico_window.title("Gráfico de Resultados Euler")
            canvas = FigureCanvasTkAgg(fig, master=grafico_window)
            canvas.draw()
            widget_canvas = canvas.get_tk_widget()
            widget_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            def on_resize(event):
                canvas.draw()

            grafico_window.bind('<Configure>', on_resize)
        else:
            tk.messagebox.showerror("Error", "No hay datos para graficar.")
    def mostrar_resultados(df):
        resultado_window = tk.Toplevel()
        resultado_window.title("Resultados de Euler")
        text = tk.Text(resultado_window)
        text.pack()
        text.insert(tk.END, df.to_string(index=False))  # Convertir DataFrame a string para mostrarlo

    euler_window = tk.Toplevel()
    euler_window.title("Método de Euler para Ecuaciones de Segundo Orden")

    tk.Label(euler_window, text="f1 (en términos de t, x, y):").pack()
    f1_entry = tk.Entry(euler_window, width=50)
    f1_entry.pack()

    tk.Label(euler_window, text="f2 (en términos de t, x, y):").pack()
    f2_entry = tk.Entry(euler_window, width=50)
    f2_entry.pack()

    tk.Label(euler_window, text="Intervalo a:").pack()
    intervalo_a_entry = tk.Entry(euler_window, width=50)
    intervalo_a_entry.pack()

    tk.Label(euler_window, text="Intervalo b:").pack()
    intervalo_b_entry = tk.Entry(euler_window, width=50)
    intervalo_b_entry.pack()

    tk.Label(euler_window, text="Tamaño de paso:").pack()
    tol_entry = tk.Entry(euler_window, width=50)
    tol_entry.pack()

    tk.Label(euler_window, text="Condición Inicial para x:").pack()
    co_x_entry = tk.Entry(euler_window, width=50)
    co_x_entry.pack()

    tk.Label(euler_window, text="Condición Inicial para y:").pack()
    co_y_entry = tk.Entry(euler_window, width=50)
    co_y_entry.pack()

    ejecutar_button = tk.Button(euler_window, text="Ejecutar", command=comando_ejecutar)
    ejecutar_button.pack()

    resultado_label = tk.Label(euler_window, text="Resultado:")
    resultado_label.pack()

    # Botón para graficar los resultados
    graficar_button = tk.Button(euler_window, text="Graficar", command=graficar_resultados)
    graficar_button.pack()

abrir_euler_2do_orden.resultado_df = None

#Runge Kutta para EDO2
def ejecutar_runge_kutta_2do_orden(f1_str, f2_str, a_val, b_val, h, co_val):
    t, x, y = symbols('t x y')
    f1_expr = sympify(f1_str.replace('np.', ''))
    f2_expr = sympify(f2_str.replace('np.', ''))

    f1_func = lambdify((t, x, y), f1_expr, modules='numpy')
    f2_func = lambdify((t, x, y), f2_expr, modules='numpy')

    def f(t, u):
        return np.array([f1_func(t, u[0], u[1]), f2_func(t, u[0], u[1])])

    try:
        t_values, y_values = runge_kutta(f, a_val, b_val, h, np.array(co_val))
        data = {'Time': t_values, 'f1(t)': [y[0] for y in y_values], "f2(t)": [y[1] for y in y_values]}
        resultado_df = pd.DataFrame(data)
        return resultado_df
    except Exception as e:
        return pd.DataFrame(columns=['Error'], data=[[str(e)]])

def abrir_runge_kutta_2do_orden():
    def comando_ejecutar():
        f1_str = f1_entry.get()
        f2_str = f2_entry.get()
        a_val = float(intervalo_a_entry.get())
        b_val = float(intervalo_b_entry.get())
        h = float(tol_entry.get())
        co_val_x = float(co_x_entry.get())
        co_val_y = float(co_y_entry.get())

        resultado_df = ejecutar_runge_kutta_2do_orden(f1_str, f2_str, a_val, b_val, h, [co_val_x, co_val_y])
        if not resultado_df.empty:
            mostrar_resultados(resultado_df)
            resultado_label.config(text="Ejecución exitosa")
            abrir_runge_kutta_2do_orden.resultado_df = resultado_df
        else:
            resultado_label.config(text="Error en la ejecución")

    def graficar_resultados():
        df = abrir_runge_kutta_2do_orden.resultado_df
        if df is not None and not df.empty:
            fig = Figure(figsize=(6, 4), dpi=100)
            plot = fig.add_subplot(1, 1, 1)
            plot.plot(df['Time'], df['f1(t)'], '-o', color='blue', label='x(t)')
            plot.plot(df['Time'], df['f2(t)'], '-o', color='red', label='y(t)')
            plot.set_title('Aproximación Runge-Kutta')
            plot.set_xlabel('Time')
            plot.set_ylabel('Values')
            plot.legend()

            grafico_window = tk.Toplevel()
            grafico_window.title("Gráfico de Resultados Runge-Kutta")
            canvas = FigureCanvasTkAgg(fig, master=grafico_window)
            canvas.draw()
            widget_canvas = canvas.get_tk_widget()
            widget_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            def on_resize(event):
                canvas.draw()

            grafico_window.bind('<Configure>', on_resize)
        else:
            tk.messagebox.showerror("Error", "No hay datos para graficar.")
    def mostrar_resultados(df):
        resultado_window = tk.Toplevel()
        resultado_window.title("Resultados de Runge-Kutta")
        text = tk.Text(resultado_window)
        text.pack()
        text.insert(tk.END, df.to_string(index=False))

    runge_kutta_window = tk.Toplevel()
    runge_kutta_window.title("Método de Runge-Kutta para Ecuaciones de Segundo Orden")

    tk.Label(runge_kutta_window, text="f1 (en términos de t, x, y):").pack()
    f1_entry = tk.Entry(runge_kutta_window, width=50)
    f1_entry.pack()

    tk.Label(runge_kutta_window, text="f2 (en términos de t, x, y):").pack()
    f2_entry = tk.Entry(runge_kutta_window, width=50)
    f2_entry.pack()

    tk.Label(runge_kutta_window, text="Intervalo a:").pack()
    intervalo_a_entry = tk.Entry(runge_kutta_window, width=50)
    intervalo_a_entry.pack()

    tk.Label(runge_kutta_window, text="Intervalo b:").pack()
    intervalo_b_entry = tk.Entry(runge_kutta_window, width=50)
    intervalo_b_entry.pack()

    tk.Label(runge_kutta_window, text="Tamaño de paso:").pack()
    tol_entry = tk.Entry(runge_kutta_window, width=50)
    tol_entry.pack()

    tk.Label(runge_kutta_window, text="Condición Inicial para x:").pack()
    co_x_entry = tk.Entry(runge_kutta_window, width=50)
    co_x_entry.pack()

    tk.Label(runge_kutta_window, text="Condición Inicial para y:").pack()
    co_y_entry = tk.Entry(runge_kutta_window, width=50)
    co_y_entry.pack()

    ejecutar_button = tk.Button(runge_kutta_window, text="Ejecutar", command=comando_ejecutar)
    ejecutar_button.pack()
    graficar_button = tk.Button(runge_kutta_window, text="Graficar", command=graficar_resultados)
    graficar_button.pack()

    resultado_label = tk.Label(runge_kutta_window, text="Resultado:")
    resultado_label.pack()

abrir_runge_kutta_2do_orden.resultado_df = None
# Crear la ventana principal
root = tk.Tk()
root.title("Métodos Numéricos")

# Crear un botón y añadirlo a la ventana principal
ceros_button = tk.Button(root, text="Ceros", command=open_ceros_window)
ceros_button.pack()

ecuaciones_button = tk.Button(root, text="Ecuaciones diferenciales", command=open_ecuaciones_window)
ecuaciones_button.pack()
# Ejecutar el bucle principal de la ventana
root.geometry("300x200")
root.mainloop()
