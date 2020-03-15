import pyvista as pv 
import numpy as np 

def getAxisToPlotter(p, axis=0, size=10):

    axis3D = np.eye(3)[axis] * 1.0

    arrow = pv.Arrow( start= size*axis3D, direction=axis3D, tip_length=0.5, tip_radius=0.3, shaft_radius=.05)
    axis = pv.Cylinder( radius = 0.05, direction=axis3D, center=0.5*size*axis3D, height=10.0, resolution=100 )
    p.add_mesh(arrow, color='orange')
    p.add_mesh(axis, color='orange')

    return

def addBasicsToPlotter(p):

    # create the three axes
    for i in range(3):
        getAxisToPlotter(p, i)

    return

def main():


    z_scale = 0.1
    w_sol = np.array([1,3])

    w_0 = np.arange(-2, 7, 0.25)
    w_1 = np.arange(-2, 7, 0.25)
    W_0, W_1 = np.meshgrid(w_0, w_1)
    MSE = (W_0 - w_sol[0])**2 + (W_1 - w_sol[1])**2
    l = 1
    cost = MSE + l * (W_0**2 + W_1**2)

    pos = np.where( cost == cost.min() )
    minW_0, munW_1, minCost = W_0[pos][0], W_1[pos][0], cost[pos][0]


    minPoint = pv.Sphere(radius=0.5, center= np.array([minW_0, munW_1, z_scale*minCost]))
    # minPoint = pv.Sphere(radius=0.5)

    mseGrid = pv.StructuredGrid( W_0, W_1, MSE*z_scale)
    mseGrid.point_arrays['mse'] =  MSE.flatten(order='F') 
    costGrid = pv.StructuredGrid( W_0, W_1, cost*z_scale)
    costGrid.point_arrays['cost'] =  cost.flatten(order='F') 
    diffGrid = pv.StructuredGrid( W_0, W_1, (cost - MSE)*z_scale)
    diffGrid.point_arrays['diff'] =  (cost - MSE).flatten(order='F') 

    # Create and plot structured grid
    p = pv.Plotter()
    addBasicsToPlotter(p)
    p.add_mesh( minPoint, color='blue' )
    # p.add_mesh(mseGrid, scalars='mse', opacity=0.8)
    p.add_mesh(costGrid, scalars='cost', opacity=0.8)
    # p.add_mesh(diffGrid, scalars='diff', opacity=0.8)
    cPos = p.show()
    # print(cPos)
    
    return 

if __name__ == '__main__':
    main()
