import numpy as np

def Trapecio(f, a, b, n):
    h = (b-a)/n
    s = 0
    for i in range(1,n):
        
        s+=f(a+i*h)
    I = h/2*(f(a)+2*s+f(b))
    return I

def simpson13(f, a, b, n):
    if(n%2 == 0):
        h=(b-a)/n
    
        Si = 0
        Sp = 0
        for i in range(1,n):
            if(i % 2 == 0):
                Sp += f(a+i*h)
            else:
                Si += f(a+i*h)
        I = (h/3)*(f(a)+2*Sp+4*Si+f(b))
        return I
    else:
        print("Esta regla requiere un numero par")
        
def simpson38(f, a, b, n):
    if(n%3 == 0):
        h=(b-a)/n
    
        Sm = 0
        Sn = 0
        for i in range(1,n):
            if(i % 3 == 0):
                Sm += f(a+i*h)
            else:
                Sn += f(a+i*h)
        I = (3*h/8)*(f(a)+2*Sm+3*Sn+f(b))
        return I
    else:
        print("Esta regla requiere un multiplo de 3")