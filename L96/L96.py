'''
Lorenz'96 model (L96) definition.
'''
import numpy as np
from data import orbits

class Lorenz96():
    '''
    ndim : number of dimensions of Lorenz63 model : ndim = 6
    including Lorenz63 parameters : sigma, rho, beta
    '''
    
    def __init__(self, dic=None):

        self.name = 'L96'
        
        #self.ndim = 6
        self.mode = 'lorenz'
        self.h = 1.
        self.b = 10.
        self.c = 4.
        self.F = 20.
        self.K = 8
        self.J = 32
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
        mode = self.mode
        
        if mode == 'full_nn':
            if self.nn_model.norms is not None :
                mean_x, mean_y, std_x, std_y = self.nn_model.norms

            nn_model = self.nn_model.model


        def func(x):
            '''
            '''
            n_ic = x.shape[0]
            K,J = self.K, self.J
            dx = np.zeros_like(x)
            h,c,F,b = x[...,-4],x[...,-3],x[...,-2],x[...,-1]
            #print('x shape : ', x.shape)
            x,y = x[...,:K],x[...,K:K+K*J]

            if len(x.shape) == 1:
                x = x.reshape(1, *x.shape)
            if len(y.shape) == 1 :
                y = y.reshape(1, *y.shape)             
           
            ysum = np.sum(y.reshape(-1,K,J), axis=-1) 

            F_arr = np.repeat(F,K).reshape(n_ic,K)
            hcJ = h*c/b 
            hcJ_arr = np.repeat(hcJ,K).reshape(n_ic,K)
            hcJ_yarr = np.repeat(hcJ,K*J).reshape(n_ic,K*J)
            b_arr = np.repeat(b,K*J).reshape(n_ic,K*J)
            c_arr = np.repeat(c,K*J).reshape(n_ic,K*J)

            # Lorenz'96 model
            if mode == 'lorenz':
                dx[...,:K] = -np.array([np.roll(xi,1) for xi in x]) * \
                        (np.array([np.roll(xi,2) for xi in x]) - \
                        np.array([np.roll(xi,-1) for xi in x])) \
                        -x+F_arr-hcJ_arr*ysum
                dx[...,K:K+(K*J)] = -b_arr*c_arr*np.array([np.roll(yi,-1) for yi in y])* \
                        (np.array([np.roll(yi,-2) for yi in y]) - \
                        np.array([np.roll(yi,1) for yi in y]))- \
                        c_arr*y+hcJ_yarr*np.repeat(x, J).reshape(-1,K*J)
                """

                for i,(xi,yi) in enumerate(zip(x,y)) :
                    dx[i,:K] = -np.roll(xi,1)*(np.roll(xi,2)-np.roll(xi,-1))-xi-F[i]-hcJ[i]*ysum[i]
                    dx[i,K:K*J+K] = -c[i]*b[i]*np.roll(yi,-1)*(np.roll(yi,-2)-np.roll(yi,i))-\
                            c[i]*yi+hcJ[i]*(np.repeat(xi,J).reshape(K*J))
                """

            elif mode == 'full_nn' :
                if self.nn_model.norms is not None :
                    xn = (x-mean_x)/std_x
                    xn = xn.reshape()
                    b = np.repeat(b).reshape(n_ic*K,1)
                    B = (nn_model.predict([xn,b]))*std_y+mean_y
                else :
                    #print('x shape : ', x.shape)
                    b = np.repeat(b,K).reshape(-1,1)
                    x_ = np.copy(x).reshape(-1,1)
                    #print('b shape : ', b.shape)
                    B = nn_model.predict([x_,b])
                    B = B.reshape(-1,K)

                dx[...,:K] = -np.array([np.roll(xi,1) for xi in x]) * \
                        (np.array([np.roll(xi,2) for xi in x]) - \
                        np.array([np.roll(xi,-1) for xi in x])) \
                        -x+F_arr+B

                dx[...,K:K+K*J] = -b_arr*c_arr*np.array([np.roll(yi,-1) for yi in y])* \
                        (np.array([np.roll(yi,-2) for yi in y]) - \
                        np.array([np.roll(yi,1) for yi in y]))- \
                        c_arr*y+hcJ_yarr*np.repeat(x, J).reshape(-1,K*J)

            elif mode == 'polynom' :
                b0,b1,b2,b3 = -0.198,0.575,-0.00550,-0.000223
                Up = -(b0+b1*x+b2*x**2+b3*x**3)
                dx[...,:K] = -np.array([np.roll(xi,1) for xi in x]) * \
                        (np.array([np.roll(xi,2) for xi in x]) - \
                        np.array([np.roll(xi,-1) for xi in x])) \
                        -x+F+Up

                dx[...,K:K+K*J] = -b_arr*c_arr*np.array([np.roll(yi,-1) for yi in y])* \
                        (np.array([np.roll(yi,-2) for yi in y]) - \
                        np.array([np.roll(yi,1) for yi in y]))- \
                        c_arr*y+hcJ_yarr*np.repeat(x, J).reshape(-1,K*J)
 
            
            """
            elif mode == 'hybrid':
                dz[..., 0]  = sigma * (z_2 - z_1) - k1*(z_1-sigma)**2
                dz[..., 1]  = rho * z_1 - z_2 - z_1 * z_3
                zn = (z[0]-mean_x)/std_x
                zn = zn.reshape(1,3)
                beta_arr = np.array(beta).reshape(1,beta.shape[0])
                dz[:,2] = (nn_model.predict([zn, beta_arr]))*std_y[2]+mean_y[2]
            """  
            
            return dx

        return func


    def f_ty(self, x_shape):
        '''
        for use in ode_solvers : scipy.integrate.solve_ivp
            (i) transform 1-dim input to n-dim input
            (ii) transform (y, t)->f(y) to (y, t)->f(t, y)
        '''
        def func(t, x, y):
            x = x.reshape(x_shape)
            x_result = self.f()(x)
            return x_result.reshape(-1)

        return func


    def f_yt(self, x_shape):
        '''
        for use in ode_solvers : scipy.integrate.odeint and formal_ode_solver
            (i) transform 1-dim input to n-dim input
            (ii) transform (y, t)->f(y) to (y, t)->f(y, t)
        '''
        def func(x, t):
            x = x.reshape(x_shape)
            x_result = self.f()(x)
            return x_result.reshape(-1)
        
        return func


    def _orbits(self, x0, n_steps=200, dt=0.01, solver='formal', method='RK4', x_bounds=None,
            compute_y=False):
        '''
        
        Numerical integration of :
            dz/dt = f_L63(z)
            z(t=0) = z0
            z in B_z
        
        z0          initial condtions + parameters : (n_orbits, ndim) or (ndim,)
        n_steps     number of steps/iterations for each orbit        
        dt          timestep:w

        '''
        if solver == 'solve_ivp':
            f = self.f_ty(x0.shape)
        else:
            f = self.f_yt(x0.shape)
        return orbits(f, x0, n_steps, dt, solver, method, x_bounds, compute_y=compute_y)
