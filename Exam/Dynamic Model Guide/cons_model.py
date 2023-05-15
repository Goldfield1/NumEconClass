import numpy as np
from types import SimpleNamespace

# optimizing and interpolation
from scipy import optimize
from scipy import interpolate


class ConsModelExam():
    def __init__(self):
        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()
        sim = self.sim = SimpleNamespace()

        # b. parameters
        par.rho = 3
        par.kappa = 0.5
        par.nu = 0.1
        par.r = 0.04
        par.beta = 0.95
        par.Delta = 0.5

        par.max_debt = (1-par.Delta)/(1+par.r)

        
        # Type 1
        par1_prb = {'low': 0.9, 'high': 0.1}
        # Type 2
        par2_prb = {'low': 0.1, 'high': 0.9}

        par.prb = {1: par1_prb, 2: par2_prb}

        # c. grids
        Nm = 500
        
        par.types = [1,2]
        par.m_grid = np.linspace(1e-8, 5, Nm)

        # d. solution arrays
        shape = (2, 2, Nm)

        # create solution grids to have the shape (2,2,500)
        # 2 points for each period
        # 2 points for each "type"
        # 500 grid points
        sol.v = np.empty(shape = shape)
        sol.c = np.empty(shape = shape)

        # e. simulation
        np.random.seed(2021)
        par.simN = 100000

        # shape (2,simN)
        # 2 for the number of periods
        # each individual in the sim has a type
        sim.m = np.empty(shape=(2,par.simN))
        sim.c = np.empty(shape=(2,par.simN))

        # set intial wealth
        sim.m[0,:] = np.fmax(np.random.normal(1, 1, size = par.simN), 0) # No one gets negative m in first period
        # set intial types, baseline everyone is type 1
        sim.types = np.ones(par.simN, dtype = int) # not dependent on time, has to be interger


 
    def utility(self, c):
        par = self.par 

        return c**(1-par.rho)/(1-par.rho)

    def bequest(self, m, c):
        par = self.par 

        return par.nu*(m-c+par.kappa)**(1-par.rho)/(1-par.rho)

    def v2(self, c2, m2):
        par = self.par

        return self.utility(c2) + self.bequest(m2, c2)

    def v1(self, _type, c1, m1, v2_interp):
        par = self.par

        # a. v2 if low income realization
        m2_low = (1+par.r)*(m1-c1) + 1-par.Delta
        v2_low = v2_interp([m2_low])[0]

        # b. v2 if high income realization
        m2_high = (1+par.r)*(m1-c1) + 1+par.Delta
        v2_high = v2_interp([m2_high])[0]

        # c. Expected v2 value
        expected_v2 = par.prb[_type]['low']*v2_low + par.prb[_type]['high']*v2_high

        # d. Total value
        return self.utility(c1) + par.beta*expected_v2

    def solve_period_2_old(self):
        par = self.par
        sol = self.sol
    
        # b. Solve consumption problem for each m2 in grid
        for i,m2 in enumerate(par.m_grid):

            # i. Objective function
            obj = lambda x: -self.v2(x[0], m2)

            # ii. Initial guess (consume half of m2)
            x0 = m2 + 0.0001

            # iii. Optimize the objective of allocating between consumption and bequests
            result = optimize.minimize(obj, [x0], method='L-BFGS-B', bounds=((1e-8, m2),))

            # iv. Save solution
            # both types have the same solution in period 2
            # type 1 (remember index starts from 0 always)
            sol.v[1, 0, i] = -result.fun
            sol.c[1, 0, i ] = result.x
            # type 2 
            sol.v[1, 1, i] = -result.fun
            sol.c[1, 1, i ] = result.x

    def solve_period_2(self):
        sol = self.sol

        # a. Solve consumption problem for each m2 in grid
        for i,m2 in enumerate(self.par.m_grid):

            # i. Objective function
            obj = lambda x: -self.v2(x[0], m2)

            # ii. Initial guess (consume half of m2)
            x0 = m2/1.1

            # iii. Optimize the objective of allocating between consumption and bequests
            result = optimize.minimize(obj, [x0], method='L-BFGS-B', bounds=((1e-12, m2),))

            # iv. Save solution
            # type 1
            sol.v[1, 0, i] = -result.fun
            sol.c[1, 0, i ] = result.x
            # type 2 
            sol.v[1, 1, i] = -result.fun
            sol.c[1, 1, i ] = result.x


    def solve_period_1(self, v2_interp):   
        par = self.par
        sol = self.sol
         
        # b. Solve for each m1 in the grid
        for j,_type in enumerate(par.types):
            for i, m1 in enumerate(par.m_grid):

                # i. Objective function
                def obj(x): return -self.v1(_type, x[0], m1, v2_interp)

                # ii. Initial guess (consume half of m1)
                x0 = m1 + 0.0001

                # iii. Optimize the objective given debt constraint
                result = optimize.minimize(obj, [x0], 
                                        method='L-BFGS-B', bounds=((1e-12, m1 + par.max_debt),))

                # iv. Save solution
                # i indexes over point in the grid
                # j indexes over the specific type
                sol.v[0,j,i] = -result.fun
                sol.c[0,j,i] = result.x[0]


    def solve(self):  
        par = self.par
        sol = self.sol
        
        # a. solve period 2
        self.solve_period_2()    
        #sol2 = solution_to_dict(sol2)

        # b. construct interpolator
        v2_interp = interpolate.RegularGridInterpolator([par.m_grid], sol.v[1,0,:],
                                                        bounds_error=False, fill_value=None)

        # c. solve period 1
        sol1 = self.solve_period_1(v2_interp)

    def simulate(self, type_to_sim):
        par = self.par
        sol = self.sol
        sim = self.sim

        # a. model has to be solved already

        # b. Construct interpolaters between cash and consumption choices
        # we have to interpolate, since ACTUAL levels of wealth may not lie on the grid
        c1_interp = interpolate.RegularGridInterpolator([par.m_grid], sol.c[0,type_to_sim-1,:],
                                                        bounds_error=False, fill_value=None)

        c2_interp = interpolate.RegularGridInterpolator([par.m_grid], sol.c[1,type_to_sim-1,:],
                                                        bounds_error=False, fill_value=None)
        # c. Simulate period 1 based on array of m and solution
        sim.c[0] = c1_interp(sim.m[0])
        sim_a1 = sim.m[0] - sim.c[0]

        # d. Transition to period 2 cash-on-hand based on random draws of income and period 1 choices
        y2_low = 1-par.Delta
        y2_high = 1+par.Delta
        y2 = np.random.choice([y2_low, y2_high], 
                            p=[par.prb[type_to_sim]['low'], par.prb[type_to_sim]['high']], 
                            size=(sim.c[1].shape))

        sim.m[1] = (1+par.r)*sim_a1 + y2

        # e. sim period 2 consumption choice based on model solution and sim_m2
        sim.c[1]  = c2_interp(sim.m[1])

        
    def simulate_good(self):
        par = self.par
        sol = self.sol
        sim = self.sim

        # a. model has to be solved already
        # doing the draws now should be faster
        y2_low = 1-par.Delta
        y2_high = 1+par.Delta
        ys_typ1 = np.random.choice([y2_low, y2_high], p=[par.prb[1]['low'], par.prb[1]['high']], size=par.simN)
        ys_typ2 = np.random.choice([y2_low, y2_high], p=[par.prb[2]['low'], par.prb[2]['high']], size=par.simN)
        ys = {1: ys_typ1, 2: ys_typ2}

        for t in range(0,2):
            for n in range(0,par.simN):
                _type = sim.types[n]

                # b. Construct interpolaters between cash and consumption choices
                # we have to interpolate, since ACTUAL levels of wealth may not lie on the grid
                c_interp = interpolate.RegularGridInterpolator([par.m_grid], sol.c[t,_type-1,:],
                                                                bounds_error=False, fill_value=None)
                
                #print(sim.m[t,n])
                # c. Simulate period 1 based on array of m and solution
                sim.c[t,n] = c_interp([sim.m[t,n]])[0] # index 0 since we only have one element to get out

                if t < 1:
                    sim_a1 = sim.m[t,n] - sim.c[t,n]

                    # d. Transition to period 2 cash-on-hand based on random draws of income and period 1 choices
                    y2 = ys[_type][n]

                    sim.m[t+1,n] = (1+par.r)*sim_a1 + y2
