import pyvista as pv 
import numpy as np 

def getAxisToPlotter(p, axis=0, size=10):

    axis3D = np.eye(3)[axis] * 1.0

    arrow = pv.Arrow( start= size*axis3D, direction=axis3D, tip_length=0.5, tip_radius=0.3, shaft_radius=.05)
    axis1 = pv.Cylinder( radius = 0.05, direction=axis3D, center=0.5*size*axis3D, height=10.0, resolution=100 )
    p.add_mesh(arrow, color='orange')
    p.add_mesh(axis1, color='orange')

    if axis in [0,2]:
        arrow = pv.Arrow( start= size*axis3D*-1, direction=axis3D*-1, tip_length=0.5, tip_radius=0.3, shaft_radius=.05)
        axis1 = pv.Cylinder( radius = 0.05, direction=axis3D, center=0.5*size*axis3D*-1, height=10.0, resolution=100 )
        p.add_mesh(axis1, color='orange')
        p.add_mesh(arrow, color='orange')


    return

def addBasicsToPlotter(p, n=3):

    # create the three axes
    for i in range(n):
        getAxisToPlotter(p, i)



    return

def generateRandomPath(startX, stopX, anchorAtStart=True, anchorYPos=0, r=0.5,scalars = None):

    if scalars is None:
        nPts = 100
    else:
        nPts = len(scalars)

    xPts = np.linspace(startX, stopX, nPts)
    zPts = np.zeros(nPts)
    yPts = np.cumsum((np.random.rand(nPts)-0.5)*r)
    if anchorAtStart:
        yPts = yPts - yPts[0] + anchorYPos
    else:
        yPts = yPts - yPts[-1] + anchorYPos


    points = np.column_stack((xPts, yPts, zPts))
    if scalars is None:
        scalars = np.ones(nPts)

    path = pv.PolyData()
    path.points = points
    cells = np.arange(0, len(points), dtype=np.int)
    cells = np.insert(cells, 0, len(points))
    path.lines = cells
    
    path["cgi"] = scalars

    tube = path.tube( radius=0.1, scalars='cgi', radius_factor=10 )


    return tube

def generateRandomSinPath(startX, stopX, anchorAtStart=True, anchorYPos=0, r=0.5, sinR= 3,scalars = None, useOld=True):

    if scalars is None:
        nPts = 100
    else:
        nPts = len(scalars)

    xPts = np.linspace(startX, stopX, nPts)
    zPts = np.zeros(nPts)
    yPts = np.sin(xPts)*sinR
    yPts += np.cumsum( (np.random.rand(nPts) - 0.5)*r )
    if anchorAtStart:
        yPts = yPts - yPts[0] + anchorYPos
    else:
        yPts = yPts - yPts[-1] + anchorYPos

    if useOld:
        yPts = np.load('../images/yPts.npy')
    else:
        np.save('../images/yPts.npy', yPts)

    points = np.column_stack((xPts, yPts, zPts))
    if scalars is None:
        scalars = np.ones(nPts)

    path = pv.PolyData()
    path.points = points
    cells = np.arange(0, len(points), dtype=np.int)
    cells = np.insert(cells, 0, len(points))
    path.lines = cells
    
    path["cgi"] = scalars

    tube = path.tube( radius=0.1, scalars='cgi', radius_factor=10 )


    return tube

def generatePutOption(p, size=6):

    axis3D = np.array([1,0,0])
    yPos = -1

    arrow1 = pv.Arrow( start= ([size-1., yPos, 0]), direction=axis3D, tip_length=0.5, tip_radius=0.15, shaft_radius=.025)
    arrow2 = pv.Arrow( start= ([1, yPos, 0]), direction=axis3D*-1, tip_length=0.5, tip_radius=0.15, shaft_radius=.025)
    axis1  = pv.Cylinder( radius = 0.025, direction=axis3D, center=np.array([0.5*size, yPos, 0]), height=size, resolution=100 )
    
    p.add_mesh(arrow1, color='#aeb6bf')
    p.add_mesh(arrow2, color='#aeb6bf')
    p.add_mesh(axis1, color='#aeb6bf')


    return

def multiPath():


    knownPath = generateRandomPath(
                -10, 0, 
                anchorAtStart=False, anchorYPos=5, 
                r=1, scalars = None)

    unknownPaths = []
    for i in range(10):
        unknownPaths.append(
            generateRandomPath(
                    0, 10, 
                    anchorAtStart=True, anchorYPos=5, 
                    r=1, scalars = None)
                )

    
    # pv.set_plot_theme('document')

    cPos = [(22.51860210286206, 27.416261394371915, 22.066261400332372),
                (0.4523407025296695, 5.3499999940395355, 0.0),
                (0.0, 0.0, 1.0)]

    p = pv.Plotter(window_size=(1000, int(1000/1.618)),
        polygon_smoothing=True,
        point_smoothing=True,
        line_smoothing=True,)
    
    addBasicsToPlotter(p, 2)
    p.add_mesh( knownPath, color='#3498db' )
    for unknownPath in unknownPaths:
        p.add_mesh( unknownPath, color='#af7ac5', opacity=0.2 )

    p.add_point_labels([(-11,0,0), (13,0,0), (0,12,0)], [ 'past', 'future', 'price'], 
        font_family='times', font_size=40, fill_shape=False, shape=None, 
        bold=False, text_color='orange',
        show_points=False)

    cPos = p.show(cpos = cPos, screenshot='../images/known_unknown.png')
    print(cPos)
    
    return 

def probabilities():


    knownPath = generateRandomPath(
                -10, 0, 
                anchorAtStart=False, anchorYPos=5, 
                r=1, scalars = None)

    unknownPath = generateRandomSinPath(
                    0, 10, 
                    anchorAtStart=True, anchorYPos=5, 
                    r=1, sinR = 2, scalars = None , useOld=True)

    
    # pv.set_plot_theme('document')

    cPos = [(22.51860210286206, 27.416261394371915, 22.066261400332372),
                (0.4523407025296695, 5.3499999940395355, 0.0),
                (0.0, 0.0, 1.0)]

    p = pv.Plotter(window_size=(1000, int(1000/1.618)),
        polygon_smoothing=True,
        point_smoothing=True,
        line_smoothing=True,)
    
    addBasicsToPlotter(p, 2)

    generatePutOption(p, size=6)

    p.add_mesh( knownPath, color='#3498db' )
    p.add_mesh( unknownPath, color='#af7ac5', opacity=0.5 )

    p.add_point_labels([(-11,0,0), (13,0,0), (0,12,0)], [ 'past', 'future', 'price'], 
        font_family='times', font_size=40, fill_shape=False, shape=None, 
        bold=False, text_color='orange',
        show_points=False, point_size=0, point_color=(0.3,0.3,0.3))

    cPos = p.show(cpos = cPos, screenshot='../images/probs.png')
    print(cPos)
    
    return 


if __name__ == '__main__':
    # multiPath()
    probabilities()
