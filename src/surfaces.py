import pyvista as pv 
import numpy as np 

def main():


    z_scale = 0.2
    w_sol = np.array([1,3])

    w_0 = np.arange(-5, 5, 0.25)
    w_1 = np.arange(-5, 5, 0.25)
    W_0, W_1 = np.meshgrid(w_0, w_1)
    MSE = (W_0 - w_sol[0])**2 + (W_1 - w_sol[1])**2

    # Create and plot structured grid
    p = pv.Plotter()
    grid = pv.StructuredGrid( W_0, W_1, MSE*z_scale)
    grid.point_arrays['mse'] =  MSE.flatten(order='F') 
    p.add_mesh(grid, scalars='mse')
    cPos = p.show()
    print(cPos)
    
    return 

if __name__ == '__main__':
    main()
