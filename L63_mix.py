'''
Lorenz'63 model (L63) definition.
'''

import numpy as np
from data import orbits

class Lorenz63():
    '''
    ndim : number of dimensions of Lorenz63 model : ndim = 6
    including Lorenz63 parameters : sigma, rho, beta
    '''
    
    def __init__(self, dic=None):

        self.name = 'L63'
        
        self.ndim = 6
        self.mode = 'lorenz'
        self.nn_model = None
        
        self.set_params(dic)


    def set_params(self, dic):
        if dic is not None:
            for key, val in zip(dic.keys(), dic.values()):
                if key in self.__dict__.keys():
                    self.__dict__[key] = val

    def f(self):
        '''
        '''
        ndim = self.ndim        
        mode = self.mode
        
        if mode != 'lorenz':
            mean_x, mean_y, std_x, std_y = self.nn_model.norms
            nn_model = self.nn_model.model

        def func(z):
            '''
            '''

            if len(z.shape) == 1:
                z = z.reshape(1, *z.shape)
            
            dz = np.zeros_like(z)
            
            z_1, z_2, z_3 = z[..., 0], z[..., 1], z[..., 2]
            sigma, rho, beta = z[..., 3], z[..., 4], z[..., 5]
            z = z[..., :3]

            # Lorenz'63 model
            if mode == 'lorenz':
                dz[..., 0]  = sigma * (z_2 - z_1) 
                dz[..., 1]  = rho * z_1 - z_2 - z_1 * z_3
                dz[..., 2]  = z_1 * z_2 - beta * z_3

            elif mode == 'full_nn':
                zn = (z[0]-mean_x)/std_x
                zn = zn.reshape(1,3)
                beta_arr = np.array(beta).reshape(1,beta.shape[0])
                dz = (nn_model.predict([zn, beta_arr]))*std_y+mean_y

            elif mode == 'hybrid':
                dz[..., 0]  = sigma * (z_2 - z_1) 
                dz[..., 1]  = rho * z_1 - z_2 - z_1 * z_3
                zn = (z[0]-mean_x)/std_x
                zn = zn.reshape(1,3)
                beta_arr = np.array(beta).reshape(1,beta.shape[0])
                dz[:,2] = (nn_model.predict([zn, beta_arr]))*std_y[2]+mean_y[2]
            
            return dz

        return func


    def f_ty(self, shape):
        '''
        for use in ode_solvers : scipy.integrate.solve_ivp
            (i) transform 1-dim input to n-dim input
            (ii) transform (y, t)->f(y) to (y, t)->f(t, y)
        '''
        def func(t, y):
            y = y.reshape(shape)
            result = self.f()(y)
            return result.reshape(-1)
        
        return func


    def f_yt(self, shape):
        '''
        for use in ode_solvers : scipy.integrate.odeint and formal_ode_solver
            (i) transform 1-dim input to n-dim input
            (ii) transform (y, t)->f(y) to (y, t)->f(y, t)
        '''
        def func(y, t):
            y = y.reshape(shape)
            result = self.f()(y)
            return result.reshape(-1)
        
        return func


    def _orbits(self, z0, n_steps=200, dt=0.01, solver='formal', method='RK4', x_bounds=None):
        '''
        
        Numerical integration of :
            dz/dt = f_L63(z)
            z(t=0) = z0
            z in B_z
        
        z0          initial condtions + parameters : (n_orbits, ndim) or (ndim,)
        n_steps     number of steps/iterations for each orbit        
        dt          timestep

        '''
        if solver == 'solve_ivp':
            f = self.f_ty(z0.shape)
        else:
            f = self.f_yt(z0.shape)
        return orbits(f, z0, n_steps, dt, solver, method, x_bounds)
