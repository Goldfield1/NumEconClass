import numpy as np
import scipy as sp
from scipy import linalg
from scipy import interpolate
from scipy import optimize
import sympy as sm
import math
from math import log
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib import cm



# Malthusian model

def u_func(model,c,n):
    """ class utility function returns utility for given 
    consumption level and number of children
    
    Args:
        
        model: class attributes
        c: consumption
        n: number of children
         
    Returns:
    
        utility
    
    """
    return (1-model.gamma)*math.log(c)+model.gamma*math.log(n) 




def solve(model):
    """solve function minimzes negative utility and saves 
    optimal values of consumption and number of children   
    
    Args:
        
        model: class attributes
    Returns (saves):
    
        model.c: optimal consumption of class
        model.n: optimal number of children
        model.u: utility for optimal consumption and number of children
    
    """
    # a. objective function (to minimize) 
    obj = lambda x: -model.u_func(x[0],x[1]) # minimize -> negtive of utility
        
    # b. constraints and bounds
    budget_constraint = lambda x: model.y-x[0]-model.rho*x[1]# violated if negative
    constraints = ({'type':'ineq','fun':budget_constraint})
        
    # c. call solver
    x0 = [model.y/2,model.y/2]    #initial guess
    sol = optimize.minimize(obj,x0,method='SLSQP',constraints=constraints)
        
    # d. save
    model.c = sol.x[0]
    model.n = sol.x[1]
    model.u = model.u_func(model.c,model.n)
    
    
class Modelproject:
    """Creates class for household problem, 
    initiates default attributes as stated in question 1,
    includes functions defined above """


    def __init__(self):
        
        # set parameter values
        self.gamma = 0.7
        self.rho = 1.1
        self.y = 100
            
    solve = solve
    u_func = u_func


# Extended model: Transition to sustained economic growth
    
def u_func_ext(model_ext,c,n,e):
    """ class extended utility function returns utility for given 
    consumption level, number of children, and education level of children
    
    Args:
        
        model: class attributes
        c: consumption
        n: number of children
        e: education level of children
         
    Returns:
    
        utility
    
    """
    return (1-model_ext.gamma)*math.log(c)+model_ext.gamma*(math.log(n) + model_ext.beta*math.log(e/(e+model_ext.g)))

def solve_ext(model_ext):
    """solve function minimzes negative utility and saves 
    optimal values of consumption, number of children, and education level  
    
    Args:
        
        model: class attributes
        
    Returns (saves):
    
        model_ext.c: optimal consumption of class
        model_ext.n: optimal number of children
        model_ext.e: optimal education level
        model_ext.u: utility for optimal consumption, number of children
    
    """
    # a. objective function (to minimize) 
    obj_ext = lambda x: -model_ext.u_func_ext(x[0],x[1],x[2]) # minimize -> negtive of utility
        
    # b. constraints and bounds
    budget_constraint_ext = lambda x: model_ext.y-x[0]-(model_ext.tau_q + model_ext.tau_e*x[2])*x[1]*model_ext.y # violated if negative
    constraints_ext = ({'type':'ineq','fun':budget_constraint_ext})
        
    # c. call solver
    x0 = [model_ext.y/3,model_ext.y/3, model_ext.y/3]    #initial guess
    sol_ext = optimize.minimize(obj_ext,x0,method='SLSQP',constraints=constraints_ext)
        
    # d. save
    model_ext.c = sol_ext.x[0]
    model_ext.n = sol_ext.x[1]
    model_ext.e = sol_ext.x[2]
    model_ext.u = model_ext.u_func_ext(model_ext.c,model_ext.n,model_ext.e)


class Modelproject_ext:
    """Creates class for household problem, 
    initiates default attributes as stated in question 1,
    includes functions defined above """


    def __init__(self):
        
        # set parameter values
        self.gamma = 0.7
        self.rho = 1.1
        self.y = 100
        self.tau_q = 0.4
        self.tau_e = 0.6
        self.beta = 1
        self.g = 0.4
            
    solve_ext = solve_ext
    u_func_ext = u_func_ext

    
#bisect optimzers for model variants   
def bisect_ss_l(a,b, gamma=0.7, rho= 1.1, A=25, X=10, alpha=1/3):

    """ 
    Input: Parameters of interest, and the interval [a,b]. This determines the interval on which the function evaluates 

    Output: Steady state value of population.  

    """
    result = optimize.bisect(lambda L: L-((gamma/rho)*(A*X)**alpha)*(L**(1-alpha)), a,b, full_output=False)
    return result

def bisect_ss_y(a,b, gamma=0.7, rho= 1.1, alpha= 1/3):

    """ 
    Input. Parameters of interest, and the interval [a,b]. This determines the interval on which the function evaluates 

    Output: Steady state value of income per capita.  

    """
    result = optimize.bisect(lambda y: y-((rho/gamma)**alpha)*y**(1-alpha), a,b, full_output=False)
    return result

def bisect_ss_l1(a,b, gamma=0.7, rho= 1.1, A=25,X=10, alpha=1/3, phi=0.1):
    
    """ 
    Input. Parameters of interest, and the interval [a,b]. This determines the interval on which the function evaluates 

    Output: Steady state value of population.  
    
    """
    result = optimize.bisect(lambda L: L-(((gamma/rho)*(A*X)**alpha)*L**(alpha*(phi-1)+1)), a,b, full_output=False)
    return result

def bisect_ss_y1(a,b,gamma=0.7,rho=1.1,alpha=1/3,phi=0.1):
    
    """ 
    Input. Parameters of interest, and the interval [a,b]. This determines the interval on which the function evaluates 

    Output: Steady state value of income per capita.  
    
    """
    result = optimize.bisect(lambda y: y-(((rho/gamma)**(alpha*(phi-1))*(y)**(alpha*phi-alpha+1))), a,b, full_output=False)
    return result

def bisect_ss_l2(a,b, gamma=0.7, tau_q=0.4, g = 0.4, tau_e=0.6):
    
    """ 
    Input. Parameters of interest, and the interval [a,b]. This determines the interval on which the function evaluates 

    Output: Steady state value of population.  
    
    """
    result = optimize.bisect(lambda L: L-((gamma/tau_q + (g*tau_e*tau_q)**0.5)*L), a,b, full_output=False)
    return result

def bisect_ss_y2(a,b, gamma=0.7, tau_q=0.4, g = 0.4, tau_e=0.6, alpha=1/3):
    
    """ 
    Input. Parameters of interest, and the interval [a,b]. This determines the interval on which the function evaluates 

    Output: Steady state value of income.  
    
    """
    result = optimize.bisect(lambda y: y-(y/(gamma/tau_q + (g*tau_e*tau_q)**0.5)**alpha), a,b, full_output=False)
    return result

#simulation function for model variants
def simulate_malthus(var, model="baseline"):
    """
   
    Plots malthusian models for simulated values 
   
    Input: var (string), specifies the variable whose path is to be evaluated
    
    Output: Plot of variable path and steady state
    
    """

    if var == "pop" and model == "baseline":
        
        def simulate_malthus_l(l,y,gamma,rho,A,X,alpha,T):
            """
            Input: variables and paramters of interest

            Output: A compbination of the solow diagram and an output diagram.

            """

            # Create lists for diagonal line d and population L
            l_list = []
            d_list = []

            # Create the population movement
            for t in range(0,T):
                l_plus = ((gamma/rho)*(A*X)**alpha)*(t**(1-alpha))
                l_list.append(l_plus)

            for t in range(0,T):
                d_plus = t
                d_list.append(d_plus)

            # Steadystate
            ss = bisect_ss_l(0.1,500,gamma,rho,A,X,alpha)

            # Plot
            plt.figure(figsize=(5,5))
            plt.plot(d_list[:T], l_list[:T], label=r'$L_{t+1}=\frac{\gamma}{\rho}(AX)^\alpha L_t^{1-\alpha}$', color = 'blue')
            plt.plot(d_list[:T], d_list[:T], label='45 degree line', color = 'black')
            plt.scatter(ss, ss, c='g', linewidths=3, label='Steady State')
            plt.text(ss, ss, '({}, {})'.format(round(ss,2), round(ss,2)))
            plt.xlim(0,T)
            plt.ylim(0,T)
            plt.ylabel('$L_{t+1}$')
            plt.xlabel('$L_t$')
            plt.grid(True)
            plt.legend()

            return plt.show()

        widgets.interact(simulate_malthus_l, 
                        l     = widgets.fixed(0), 
                        y     = widgets.fixed(0),
                        gamma =  widgets.FloatSlider(description = r'$\gamma$' , min = 0 ,    max = 0.99 , step = 0.05 , value = 0.7),
                        rho   =  widgets.FloatSlider(description = r'$\rho$' , min = 0.5 ,    max = 2 , step = 0.05 , value = 1.1),
                        A     =  widgets.FloatSlider(description = '$A$' , min = 0 ,    max =50 , step = 0.5 , value = 25),
                        X     =  widgets.FloatSlider(description = '$X$' , min = 0 ,    max =50 , step = 0.5 , value = 10),
                        alpha = widgets.FloatSlider(description = r'$\alpha$' , min = 0 ,    max = 0.99 , step = 0.05 , value = 1/3),

                        T     = widgets.IntSlider(description='$T$' ,          min = 0,     max = 250, step = 1,    value = 100))
        
    elif var == "inc" and model == "baseline":
        
        def simulate_malthus_y(l,y,gamma,rho,alpha,T):
            """
            Input: varibales and paramters of interest

            Output: A compbination of the solow diagram and an output diagram.

            """

            # Create lists for diagonal line d and y
            y_list = []
            d_list = []

            # Create linspace for computation of non-integer steps
            step_bin = np.linspace(0, T, T*10+1) 

            # Create the income per capita movement
            for i in range(len(step_bin)):
                t = step_bin[i]
                y_plus = ((rho/gamma)**alpha)*t**(1-alpha)
                y_list.append(y_plus)

            for i in range(len(step_bin)):
                d_plus = step_bin[i]
                d_list.append(d_plus)

            ss = bisect_ss_y(0.1,500,gamma,rho,alpha)
            # Plot
            plt.figure(figsize=(5,5))
            plt.plot(d_list[:T*10+1], y_list[:T*10+1], label=r'$\frac{\gamma}{\rho}(AX)^\alpha L^{1-\alpha}$', color = 'blue')
            plt.plot(d_list[:T*10+1], d_list[:T*10+1], label='Diagonal', color = 'black')
            plt.scatter(ss, ss, c='g', linewidths=3, label='Steady State')
            plt.text(ss, ss, '({}, {})'.format(round(ss,2), round(ss,2)))
            plt.xlim(0,T)
            plt.ylim(0,T)
            plt.ylabel('$y_{t+1}$')
            plt.xlabel('$y_t$')
            plt.grid(True)
            plt.legend()

            return plt.show()

        widgets.interact(simulate_malthus_y, 
                        l     = widgets.fixed(0), 
                        y     = widgets.fixed(0),
                        gamma =  widgets.FloatSlider(description = r'$\gamma$' , min = 0 ,    max = 0.99 , step = 0.05 , value = 0.7),
                        rho   =  widgets.FloatSlider(description = r'$\rho$' , min = 0.5 ,    max = 2 , step = 0.05 , value = 1.1),
                        alpha = widgets.FloatSlider(description = r'$\alpha$' , min = 0 ,    max = 0.99 , step = 0.05 , value = 1/3),

                        T     = widgets.IntSlider(description='$T$' ,          min = 1,     max = 50, step = 1,    value = 10                                         ))
        
    elif var == "pop" and model == "ext1":
    
        def simulate_malthus_l1(l,y,gamma,rho,A,X,alpha,phi,T):
            """
            Input: varibales and paramters of interest

            Output: A compbination of the solow diagram and an output diagram.

            """

            # a. create lists, diagonal line d, and population l
            l_list = []
            d_list = []
            # b. create the population movement
            for t in range(0,T):
                l_plus = (((gamma/rho)*(A*X)**alpha)*t**(alpha*(phi-1)+1)) 
                l_list.append(l_plus)


            for t in range(0,T):
                d_plus = t
                d_list.append(d_plus)

            # c. steadystate
            ss = bisect_ss_l1(0.1,500,gamma,rho,A,X,alpha,phi)

            # p- plot
            plt.figure(figsize=(5,5))
            plt.plot(l_list[:T], label=r'$L_{t+1}=\frac{\gamma}{\rho}(AX)^\alpha L_t^{\alpha(\phi-1)+1}$', color = 'blue')
            plt.plot(d_list[:T], label='45 degree line', color = 'black')
            plt.scatter(ss, ss, c='g', linewidths=3, label='Steady State')
            plt.text(ss, ss, '({}, {})'.format(round(ss,2), round(ss,2)))
            plt.xlim(0,T)
            plt.ylim(0,T)
            plt.ylabel('$L_{t+1}$')
            plt.xlabel('$L_t$')
            plt.grid(True)
            plt.legend()

            return plt.show()

        widgets.interact(simulate_malthus_l1, 
                        l     = widgets.fixed(0), 
                        y     = widgets.fixed(0),
                        gamma =  widgets.FloatSlider(description = r'$\gamma$' , min = 0 ,    max = 0.99 , step = 0.05 , value = 0.7),
                        rho   =  widgets.FloatSlider(description = r'$\rho$' , min = 0.5 ,    max = 2 , step = 0.05 , value = 1.1),
                        A     =  widgets.FloatSlider(description = '$A$' , min = 0 ,    max =50 , step = 0.5 , value = 25),
                        X     =  widgets.FloatSlider(description = '$X$' , min = 0 ,    max =50 , step = 0.5 , value = 10),
                        alpha = widgets.FloatSlider(description = r'$\alpha$' , min = 0 ,    max = 0.99 , step = 0.05 , value = 1/3),
                        phi   = widgets.FloatSlider(description = r'$\phi$' , min = 0 ,    max = 0.4 , step = 0.05 , value = 0.1),
                        T     = widgets.IntSlider(description='$T$' ,          min = 0,     max = 250, step = 1,    value = 100))
    
    elif var == "inc" and model == "ext1":

        def simulate_malthus_y1(l,y,gamma,rho,alpha,T,phi):
            """
            Input: varibales and paramters of interest

            Output: A compbination of the solow diagram and an output diagram.

            """

            # Create lists for, diagonal line d and income y
            y_list = []
            d_list = []

            step_bin = np.linspace(0, T, T*10+1)

            # Create the population movement
            for i in range(len(step_bin)):
                t = step_bin[i]
                y_plus = (((rho/gamma)**(alpha*(phi-1))*(t)**(alpha*phi-alpha+1)))
                y_list.append(y_plus)

            for i in range(len(step_bin)):
                d_plus = step_bin[i]
                d_list.append(d_plus)

            ss = bisect_ss_y1(0.1,500,gamma,rho,alpha)
            # Plot
            plt.figure(figsize=(5,5))
            plt.plot(d_list[:T*10+1], y_list[:T*10+1], label=r'$y_{t+1} = (AX)^\alpha (n_tL_t)^{\alpha(\phi-1)}$', color = 'blue')
            plt.plot(d_list[:T*10+1], d_list[:T*10+1], label='45° line', color = 'black')
            plt.scatter(ss, ss, c='g', linewidths=3, label='Steady State')
            plt.text(ss, ss, '({}, {})'.format(round(ss,2), round(ss,2)))
            plt.xlim(0,T)
            plt.ylim(0,T)
            plt.ylabel('$y_{t+1}$')
            plt.xlabel('$y_t$')
            plt.grid(True)
            plt.legend()

            return plt.show()

        widgets.interact(simulate_malthus_y1, 
                        l     = widgets.fixed(0), 
                        y     = widgets.fixed(0),
                        gamma =  widgets.FloatSlider(description = r'$\gamma$' , min = 0 ,    max = 0.99 , step = 0.05 , value = 0.7),
                        rho   =  widgets.FloatSlider(description = r'$\rho$' , min = 0.5 ,    max = 2 , step = 0.05 , value = 1.1),
                        alpha = widgets.FloatSlider(description = r'$\alpha$' , min = 0 ,    max = 0.99 , step = 0.05 , value = 1/3),
                        phi   = widgets.FloatSlider(description = r'$\phi$' , min = 0 ,    max = 0.4 , step = 0.05 , value = 0.1),
                        T     = widgets.IntSlider(description='$T$' ,          min = 0,     max = 50, step = 1,    value = 10                                         ))

    elif var == "pop" and model == "ext2":

        def simulate_malthus_l2(l,y,gamma,tau_q,tau_e,g,T):
            """
            Input: varibales and paramters of interest

            Output: A compbination of the solow diagram and an output diagram.

            """

            # a. create lists for diagonal line d, and population
            l_list = []
            d_list = []

            # b. create the population movement
            for t in range(0,T):
                l_plus = (gamma/(tau_q+(g*tau_e*tau_q)**0.5))*t
                l_list.append(l_plus)

            for t in range(0,T):
                d_plus = t
                d_list.append(d_plus)

            ss = bisect_ss_l2(0,500,gamma,tau_q,g,tau_e)

            # c. plot
            plt.figure(figsize=(5,5))
            plt.plot(l_list[:T], label=r'$L_{t+1}=\frac{\gamma}{\rho}(AX)^\alpha L_t^{1-\alpha}$', color = 'blue')
            plt.plot(d_list[:T], label='45° line', color = 'black')
            plt.scatter(ss, ss, c='g', linewidths=3, label='Steady State')
            plt.text(ss, ss, '({}, {})'.format(round(ss,2), round(ss,2)))
            #plt.scatter(pss2, pss2, c='g', linewidths=3, label='Steady State')
            #plt.text(pss2, pss2, '({}, {})'.format(round(pss2,2), round(pss2,2)))
            plt.xlim(0,T)
            plt.ylim(0,T)
            plt.ylabel('$L_{t+1}$')
            plt.xlabel('$L_t$')
            plt.grid(True)
            plt.legend()

            return plt.show()

        widgets.interact(simulate_malthus_l2, 
                        l     = widgets.fixed(0), 
                        y     = widgets.fixed(0),
                        gamma =  widgets.FloatSlider(description = r'$\gamma$' , min = 0 ,    max = 0.99 , step = 0.05 , value = 0.7),
                        tau_q   =  widgets.FloatSlider(description = r'$\tau_q$' , min = 0.5 ,    max = 2 , step = 0.05 , value = 0.4),
                        tau_e     =  widgets.FloatSlider(description = r'$\tau_e$' , min = 0 ,    max =2 , step = 0.05 , value = 0.6),
                        g    =  widgets.FloatSlider(description = '$g$' , min = 0 ,    max =0.2 , step = 0.01 , value = 0.4),
                        T     = widgets.IntSlider(description='$T$' ,          min = 0,     max = 100, step = 1,    value = 50))


    elif var == "inc" and model == "ext2":

        def simulate_malthus_y2(y,gamma,alpha,tau_q,tau_e,g,T):
            """
            Input: varibales and paramters of interest

            Output: A compbination of the solow diagram and an output diagram.

            """

            # a. create lists for diagonal line d and income y
            y_list = []
            d_list = []
            # b. create the population movement
            for t in range(0,T):
                y_plus = t/((gamma/(tau_q+(g*tau_e*tau_q)**0.5))**alpha)
                y_list.append(y_plus)      

            for t in range(0,T):
                d_plus = t
                d_list.append(d_plus)

            ss = bisect_ss_y2(0,500,gamma,tau_q,g,tau_e,alpha)
            # c. plot
            plt.figure(figsize=(5,5))
            plt.plot(y_list[:T], label=r'$y_{t+1} = \frac{y_t}{(\frac{\gamma}{\tau_q + (g \tau_e \tau_q)^{0.5}})^\alpha}$', color = 'blue')
            plt.plot(d_list[:T], label='45° line', color = 'black')
            plt.scatter(ss, ss, c='g', linewidths=3, label='Steady State')
            plt.text(ss, ss, '({}, {})'.format(round(ss,2), round(ss,2)))
            #plt.scatter(yss_y2, yss_y2, c='g', linewidths=3, label='Steady State')
            #plt.text(yss_y2, yss_y2, '({}, {})'.format(round(yss_y2,2), round(yss_y2,2)))
            plt.ylabel('$y_{t+1}$')
            plt.xlabel('$y_t$')
            plt.xlim(0,T)
            plt.ylim(0,T)
            plt.grid(True)
            plt.legend()

            return plt.show()

        widgets.interact(simulate_malthus_y2, 
                        y     = widgets.fixed(0), 
                        gamma =  widgets.FloatSlider(description = r'$\gamma$' , min = 0 ,    max = 0.99 , step = 0.05 , value = 0.7),
                        alpha = widgets.FloatSlider(description = r'$\alpha$' , min = 0 ,    max = 0.99 , step = 0.05 , value = 1/3),
                        tau_q     =  widgets.FloatSlider(description = r'$\tau_q$' , min = 0 ,    max =2 , step = 0.05 , value = 0.4),
                        tau_e     =  widgets.FloatSlider(description = r'$\tau_e$' , min = 0 ,    max =2 , step = 0.05 , value = 0.6),
                        g = widgets.FloatSlider(description = r'$g$' , min = 0 ,    max = 0.2 , step = 0.05 , value = 0.4),

                        T     = widgets.IntSlider(description='$T$' ,          min = 0,     max = 250, step = 1,    value = 100))

    else:
        print("Variable of interest or model type not specified correctly!")