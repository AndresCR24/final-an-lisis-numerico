import numpy as np
import sympy as sp

x=sp.symbols("x")
def Biseccion(f,a,b,tol):
    if f(a)*f(b)>0:
        print("La funcion no cumple el teorema del intervalo")
    else:
        while(np.abs(b-a)>tol):
            c=(a+b)/2
            if f(a)*f(c)<0:
                b=c
            else:
                a=c
    return c


def Posicion_falsa(f, a, b, tol):
    if f(a) * f(b) > 0:
        print("La función no cumple el teorema en el intervalo, busque otro intervalo")
    else:
        while True:
            c = a - ((f(a) * (a - b)) / (f(a) - f(b)))
            if np.abs(f(c)) <= tol:
                break
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c
        return c

def Newton(f, x0, tol):
    df = sp.diff(f, x)
    newton = x - f / df
    newton = sp.lambdify(x, newton)
    i = 1
    while True:
        x1 = newton(x0)
        if np.abs(x1 - x0) <= tol:
            break
        x0 = x1
        i += 1
    print('La raíz de la función es: ', x1)
    print('La cantidad de iteraciones fueron: ', i)
    return x1

"""
def Secante(f, x0, x1, tol):
    x2 = x1-f(x1)*(x1-x0)/(f(x0)-f(x1))
    while(np.abs(x2-x1)>tol):
        x0=x1
        x1=x2
        x2 = x1-f(x1)*(x0-x1)/(f(x0)-f(x1))
    print('La raiz de la funcion es: ', x2)
    return
"""
def Secante(f, x0, x1, tol):
    try:
        x2 = x1 - f(x1) * (x1 - x0) / (f(x0) - f(x1))
        while np.abs(x2 - x1) > tol:
            x0 = x1
            x1 = x2
            x2 = x1 - f(x1) * (x0 - x1) / (f(x0) - f(x1))
        print('La raiz de la funcion es: ', x2)
        return x2
    except ZeroDivisionError:
        print("Error: División por cero.")
        return None
    except TypeError as e:
        print(f"Error de tipo: {e}")
        return None
