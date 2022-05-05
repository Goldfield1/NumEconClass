import numpy as np
from scipy import optimize
import sympy as sm


def sympy_duopoly(a_low, a_high, b, c):

    """ Analytically solve the cournot model for different parameters

    Args:
        a_low (float): low a (max demand) limit
        a_high (float): high a (max demand) limit
        b (float): elasticity of substitution
        c (float): marginal cost

    Returns:
        a_list (list): list of a-values used
        q1_star_list (list): optimal quantities produced for each a-value for firm 1
        q2_star_list (list): optimal quantities produced for each a-value for firm 2
        pi_1_list (list): profits obtained for each a-value for firm 1
        pi_2_list (list): profits obtained for each a-value for firm 2

    """ 
    #define symbols, lists and parameters:
    q1 = sm.symbols("q_1")
    q1_star_list = []
    q2 = sm.symbols("q_2")
    q2_star_list = []
    pi_1_list = []
    pi_2_list = []
    a_list = []
    MC = c

    #start loop with writing up the price (as a function of q) for given a-value. appends the a-value onto list
    for a in range(a_low, a_high+1):
        p = a - (b*q1+b*q2)
        a_list.append(a)

        #derive profit-function for firm one and isolate q1 as function of q2
        obj_1 = (p-MC)*q1
        foc_1 = sm.diff(obj_1, q1)
        q1_temp = sm.solve(sm.Eq(foc_1,0), q1)
        q1_temp = q1_temp[0]

        #derive profit function, isolate q2 and substitute q1 in to get optimal q2
        obj_2 = (p-MC)*q2
        foc_2 = sm.diff(obj_2, q2)
        foc_2_subs = foc_2.subs(q1, q1_temp)
        q2_star = sm.solve(foc_2_subs, q2)
        q2_star = q2_star[0]

        #make sure that q2 is positive, else set to 0
        if q2_star < 0:
            q2_star = 0

        #substitute optimal q2 into q1 to get optimal q1
        q1_star = q1_temp.subs(q2, q2_star)

        #make sure that q1 is positive else set to 0
        if q1_star < 0:
            q1_star = 0

        #append optimal q1 and q1 onto lists
        q1_star_list.append(q1_star)
        q2_star_list.append(q2_star)

        #substitute optimal q-values into p
        p = p.subs(q1, q1_star)
        p = p.subs(q2, q2_star)

        #calculate profits obtained from optimal q-values
        pi_1 = (p - MC)*q1_star
        pi_2 = (p - MC)*q2_star

        #set profits to 0 if profits are negative (they shouldn't be as quantites have been set to 0 as above, but just to be sure)
        if pi_1 < 0:
            pi_1 = 0       
        if pi_2 < 0:
            pi_2 = 0

        #append profits onto lists
        pi_1_list.append(pi_1)
        pi_2_list.append(pi_2)

        
    return a_list,q1_star_list,q2_star_list,pi_1_list,pi_2_list


def scipy_duopoly(a,b,c):

    """ Numerically solve a cournot duopoly given parameters

    Args:
    a (float) = max demand 
    b (float) = elasticity of substitution
    c (float) = marginal cost

    Returns:
    opt_vec (list) = optimal quantities produced for both firms

    """

    #define demand
    def demand_numopt_2(q1,q2,a,b):

        """ Define demand function for two firms

        Args: 
        q1 (float) = quantity produced for firm 1
        q2 (float) = quantity produced for firm 2
        a (float) = max demand
        b (float) = elasticity of substitution

        Returns:
        a - (b*q1 + b*q2) (float) = demand function

        """

        return a - (b*q1 + b*q2)

    #define profit
    def profit_numopt_2(q1,q2,a,b,c):

        """ Define profit function for two firms

        Args: 
        q1 (float) = quantity produced for firm 1
        q2 (float) = quantity produced for firm 2
        a (float) = max demand
        b (float) = elasticity of substitution
        c (float) = marginal cost

        Returns:
        (demand_numopt_2(q1,q2,a,b)-c)*q1 (float) = profit for firm 1 (effectively any firm, since they're identical)

        """

        return (demand_numopt_2(q1,q2,a,b)-c)*q1

    #maximize profits
    def profit_max_2(q2,a,b,c):

        """ Maximize profits for one firm given the others choice

        Args:
        q2 (float) = quantity produced for other firm
        a (float) = max demand
        b (float) = elasticity of substitution
        c (float) = marginal cost

        Returns:
        q1[0] (float) = optimal quantity given other firms choice

        """

        q1 = optimize.brute(lambda q: -profit_numopt_2(q,q2,a,b,c), ((0,100,),) )

        return q1[0]

    #define the fixed point equilibrium: x - f(x) 
    def profit_max_vector_2(q_vec, a, b, c):

        """ Generate equilibrium 

        Args:
        q_vec (float) = quantity produced for both firms as vector
        a (float) = max demand
        b (float) = elasticity of substitution
        c (float) = marginal cost

        Returns:
        np.array(q_vec) - np.array( [profit_max_2(q_vec[1], a, b, c) , profit_max_2(q_vec[0], a, b, c)] ) (list) = vector consisting of optimal quantities for both firms

        """

        #second part is a list containing the profitmaximizing decision for firm 1 and firm 2 respectively; these are the best response functions
        return np.array(q_vec) - np.array( [profit_max_2(q_vec[1], a, b, c) , profit_max_2(q_vec[0], a, b, c)] )

    #generate starting guesses for solving the vector
    q_vec = np.ones(2)

    #find the fixed point equilibrium by solving the above vector
    opt_vec = optimize.fsolve(profit_max_vector_2, q_vec, args = (a,b,c))

    return opt_vec




def scipy_oligopoly(no_firms,a,b,c):

    """ Numerically solve a cournot oligopoly for a given number of firms and given parameters

    Args:
    no_firms (int) = number of firms in the oligopoly
    a (float) = max demand 
    b (float) = elasticity of substitution
    c (float) = marginal cost

    Returns:
    opt_vec (list) = optimal quantities produced for all firms

    """

    #generate q_list - used for both guesses at the final step but also to determine the length of the vectors in the next few steps
    q_list = np.ones(no_firms)

    #define demand, now with the second part as a matrix multiplication due to the vector of q's
    def demand_numopt_m(q_list,a,b):

        """ Define demand function for n number of firms 

        Args:
        q_list (array) = quantity produced for all firms as a vector
        a (float) = max demand
        b (float) = elasticity of substitution

        Returns:
        a - np.matmul(b*np.ones(len(q_list)),q_list) (float) = demand function

        """

        return a - np.matmul(b*np.ones(len(q_list)),q_list)

    #define profit - now the multiplying q is not hardcoded but depends on the firm_index
    def profit_numopt_m(q_list,firm_index,a,b,c):

        """ Define profit function n number of firms

        Args: 
        q_list (array) = quantity produced for all firms as a vector
        firm_index (float) = firm index (which firm is the profits calculated for)
        a (float) = max demand
        b (float) = elasticity of substitution
        c (float) = marginal cost

        Returns:
        (demand_numopt_m(q_list,a,b)-c)*q_list[firm_index]

        """

        return (demand_numopt_m(q_list,a,b)-c)*q_list[firm_index]

    #maximize profits
    def profit_max_m(q_list,firm_index,a,b,c):

        """ Maximize profits for one firm given the choice of the others'

        Args:
        q_list (array) = quantity produced for all firms as a vector
        firm_index (float) = firm index (which firm is the profits calculated for)
        a (float) = max demand
        b (float) = elasticity of substitution
        c (float) = marginal cost

        Returns:
        q_optimal_response = optimal quantity given the other firms' choice

        """

        #create temporary q_list so I can use one of these q's for solving given the rest
        def opt_response(q_temp):

            """ Create temporary list to use one of the q's for solving 

            Args:
            q_temp (float) = q for given firm

            Returns:
            -profit_numopt_m(q_list_temp,firm_index,a,b,c) (float) = profit maximization function

            """
            q_list_temp = q_list.copy()
            q_list_temp[firm_index] = q_temp
            return -profit_numopt_m(q_list_temp,firm_index,a,b,c)

        #solve for optimal response
        q_optimal_response = optimize.brute(opt_response, ((0,100,),) )

        return q_optimal_response

    #define fixed point equilibrium
    def profit_max_vector_m(q_list, a, b, c): 

        """ Generate equilibrium 

        Args:
        q_list (array) = quantity produced for all firms
        a (float) = max demand
        b (float) = elasticity of substitution
        c (float) = marginal cost

        Returns:
        opt_vec (list) = vector consisting of optimal quantities for all firms

        """

        #create BR-list
        best_response = []

        #for each firm index profit maximize given the other firms and append onto list (corresponding to the exact same step as above, but now the list is consolidated)
        for firm_index in range(len(q_list)):
            best_response.append(profit_max_m(q_list,firm_index,a,b,c))

        #remake to array and reshape best_response for fixed-point calculations
        best_response = np.array(best_response).reshape(-1)

        return q_list - best_response

    #calculate fixed point equilibrium by solving the above vector
    opt_vec = optimize.fsolve(profit_max_vector_m, q_list, args = (a,b,c))

    return opt_vec




def scipy_figure(no_firms_iter,a,b,c):
    """ Generate lists of optimal quantities over many different number of firms in the oligopoly model

    Args:
    no_firms_iter (int) = Chooses max number of firms the model should go through
    a (float) = max demand
    b (float) = elasticity of substitution 
    c (float) = marginal cost

    Returns: 
    opt_vec_list (list) = list of optimal quantites produced for firm 1 (same for all firms) given number of firms in the model
    no_firms_iter (list) = list of number of firms the model went through
    """


    opt_vec_list = []
    no_firms_iter_list = []

    for no_firms in range(2,no_firms_iter+1):
        no_firms_iter_list.append(no_firms)

        opt_vec_iter = scipy_oligopoly(no_firms,a,b,c)
        opt_vec_list.append(opt_vec_iter[0])

    return opt_vec_list,no_firms_iter_list