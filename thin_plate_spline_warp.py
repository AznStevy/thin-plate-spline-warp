import numpy as np

# try thin plate spline warping
def thin_plate_spline_warp(warped_pts, ctrl_pts, obj_to_warp):

    # convert everything to np array
    warped_pts = np.array(warped_pts)
    ctrl_pts = np.array(ctrl_pts)
    obj_to_warp = np.array(obj_to_warp)

    num_points = warped_pts.shape[0]
    K = np.zeros(num_points, num_points)
    for rr in np.arange(num_points):
        for cc in np.arange(num_points):
            K[rr,cc] = np.sum(np.subtract(warped_pts[rr,:], warped_pts[cc,:])) #R^2 
            K[cc,rr] = K[rr,cc]

    #calculate kernel function R
    K = np.amax(K, 1e-320) 
    #K = K.* log(sqrt(K))
    K = np.sqrt(K) #
    # Calculate P matrix
    P = [np.ones(num_points, 1), warped_pts] #nX4 for 3D
    # Calculate L matrix
    L = [[K, P][P.T, np.zeros(4,4)] ]
    param = np.pinv(L) * [ctrl_pts, np.zeros(4,3)]
    # Calculate new coordinates (x',y',z') for each points 
    #
    num_points_obj = obj_to_warp.shape[0]

    K = np.zeros(num_points_obj, num_points)
    gx = obj_to_warp[:,1]
    gy = obj_to_warp[:,2]
    gz = obj_to_warp[:,3]

    for nn in np.arange(num_points):
        K[:,nn] = np.square(np.subtract(gx, warped_pts[nn,0])) + np.square(np.subtract(gy, warped_pts[nn,2])) + np.square(np.subtract(gz, warped_pts[nn,3])) # R^2
 
    K = np.amax(K, 1e-320) 
    K = np.sqrt(K) #|R| for 3D
    P = [np.ones(num_points_obj,1), gx, gy, gz]
    L = [K, P]
    object_warped = np.matmul(L, param)  
    object_warped[:,1] = np.round(object_warped[:,1]*10^3)*10^-3
    object_warped[:,2] = np.round(object_warped[:,2]*10^3)*10^-3
    object_warped[:,3] = np.round(object_warped[:,3]*10^3)*10^-3

    return object_warped

# test tpsw
warped_pts = [[1,2,3], [4,5,6], [7,8,9]]
ctrl_pts = [[0,0,0], [1,2,3], [6,3,1]]
obj_to_warp = [[3,2,5], [3,7,3]]

thin_plate_spline_warp(warped_pts, ctrl_pts, obj_to_warp)