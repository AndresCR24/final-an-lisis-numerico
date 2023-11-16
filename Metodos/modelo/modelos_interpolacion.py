import numpy as np
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt
X = sp.symbols('X')

def p_simple(xdata, ydata):
    N = len(xdata)
    M = np.zeros([N, N])
    P = 0
    
    for i in range(N):
        M[i, 0] = 1  # Corregido: MLi,0］=1 (caracteres no válidos y error de sintaxis)
        for j in range(1, N):
            M[i, j] = M[i, j-1] * xdata[i]  # Corregido: ME1,j］=M［i,j-1］*xdata［i］ (caracteres no válidos)
    
    ai = np.linalg.solve(M, ydata)
    
    for i in range(N):
        P = P + ai[i] * X**i  # Corregido: ×**i (caracter incorrecto por X)
    
    print('El polinomio interpolante es: P(X)=', P)
    return sp.lambdify(X, P)  # Esto crea una función lambda que puede evaluar P en cualquier valor de X

def lagrange(xdata, ydata):
    N = len(xdata)
    P = 0  # Inicializar el polinomio interpolante a 0
    
    for i in range(N):
        T = 1  # Inicializar el término de Lagrange para i-ésimo término
        
        for j in range(N):
            if j != i:  # Construir el i-ésimo término de Lagrange, excluyendo el j = i
                T = T * (X - xdata[j]) / (xdata[i] - xdata[j])
        
        P = P + T * ydata[i]  # Sumar el i-ésimo término al polinomio interpolante

    # Imprimir el polinomio interpolante expandido
    print('El polinomio es P(X):', sp.expand(P))
    
    # Retornar una función lambda que evalúe el polinomio para cualquier valor de X
    return sp.lambdify(X, P)

import numpy as np
import matplotlib.pyplot as plt

def least_squares_fit(xdata, ydata, degree):
    """
    Ajusta un polinomio del grado dado a los datos utilizando el método de mínimos cuadrados.

    Parámetros:
    xdata : array_like, valores de x (independiente)
    ydata : array_like, valores de y (dependiente)
    degree : int, grado del polinomio a ajustar

    Retorna:
    p : np.poly1d, el polinomio que mejor ajusta los datos
    """
    # Crear la matriz de Vandermonde para el ajuste de mínimos cuadrados
    A = np.vander(xdata, degree + 1)
    # Resolver el sistema lineal para encontrar los coeficientes del polinomio
    coeffs = np.linalg.lstsq(A, ydata, rcond=None)[0]
    # Crear un objeto polinomial con los coeficientes encontrados
    p = np.poly1d(coeffs[::-1])

    return p

# Ejemplo de uso:
# xdata = np.array([1, 2, 3, 4, 5])
# ydata = np.array([5.5, 43.1, 128, 290.7, 498.4])
# degree = 2  # Ajustar un polinomio de grado 2

# p = least_squares_fit(xdata, ydata, degree)

# plt.scatter(xdata, ydata, label='Datos originales')
# plt.plot(xdata, p(xdata), label='Ajuste de mínimos cuadrados')
# plt.legend()
# plt.show()
def Minimos_cuadrados(x, y):
    # np.polyfit devuelve los coeficientes [a1, a0]
    a1, a0 = np.polyfit(x, y, 1)
    print(f"Intercepto (a0): {a0}")
    print(f"Pendiente (a1): {a1}")
    return a0, a1

# Función para el modelo de potencias
def modelo_potencias(x, a, b):
    return a * x**b