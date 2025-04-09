import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import streamlit as st

# Extraer los bordes del tablero
def get_chessboard_points(chessboard_shape:tuple, dx:int, dy:int) -> np.ndarray:    
    points = np.zeros((chessboard_shape[0]*chessboard_shape[1], 3), np.float32)
    y, x = np.meshgrid(np.arange(chessboard_shape[0]), np.arange(chessboard_shape[1]))
    points[:,1] = y.ravel() * dx
    points[:,0] = x.ravel() * dy

    return points

# Extraer los parametros de calibración
def calibrate_camera(imgpath: str) -> tuple:
    imglist = os.listdir(imgpath)
    imgs = [cv2.imread(imgpath+img) for img in imglist]

    rets = []
    corners = []
    for imagen in imgs:
        ret,corner = cv2.findChessboardCorners(imagen,(7,7),None)
        rets.append(ret)
        corners.append(corner)

    cb_points = get_chessboard_points((7,7), 23, 23)

    # Realizar la calibración, obtener K
    valid_corners = []
    image_points = []

    for idx, corner in enumerate(corners):
        if rets[idx] and len(corner) == 49:
            valid_corners.append(corner)
            image_points.append(corner.reshape(-1,2))
    num_valid_images = len(valid_corners)

    # Prepare input data 
    object_points = np.array([get_chessboard_points((7, 7),23, 23)] * num_valid_images, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)

    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (valid_corners[0].shape[1], valid_corners[0].shape[0]), None, None, flags=cv2.CALIB_FIX_ASPECT_RATIO)

    return intrinsics, dist_coeffs

def plot3DAxis(axes, Pts):
    # Eje X: punto en fila 0 - punto en fila 1
    x = np.array([Pts[0,0], Pts[1,0]]) 
    y = np.array([Pts[0,1], Pts[1,1]]) 
    z = np.array([Pts[0,2], Pts[1,2]]) 
    axes.plot3D(x, y, z, 'r')    

    # Eje Y: punto en fila 0 - punto en fila 2
    x = np.array([Pts[0,0], Pts[2,0]]) 
    y = np.array([Pts[0,1], Pts[2,1]]) 
    z = np.array([Pts[0,2], Pts[2,2]]) 
    axes.plot3D(x, y, z, 'g')    
    
    # Eje Z: punto en fila 0 - punto en fila 3
    x = np.array([Pts[0,0], Pts[3,0]]) 
    y = np.array([Pts[0,1], Pts[3,1]]) 
    z = np.array([Pts[0,2], Pts[3,2]]) 
    axes.plot3D(x, y, z, 'b')   

def plot3DPoints(Pts, axes):   
    x = Pts[:, 0]
    y = Pts[:, 1]
    z = Pts[:, 2]
    axes.scatter3D(x, y, z, 'k')       

def plotCamera3D(A, axes=None):   
    if axes is None:
        axes = plt.axes(projection = '3d')
    
    l = 50
    Pts = np.array([[0, 0, 0, 1], [l, 0, 0, 1], [0, l, 0, 1], [0, 0, l, 1]])

    Pts = np.dot(A, Pts.T).T
    Pts[:, 2] *= -1
    plot3DAxis(axes, Pts)
    
    # Set the labels for the axes
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    
    # Set the limits for the axes
    X = Pts[:, 0]
    Y = Pts[:, 1]
    Z = Pts[:, 2]
    axes.auto_scale_xyz(X, Y, Z)

def image_augmentation(frame, src_image, dst_points):
    src_h, src_w = src_image.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    src_points = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_w]])
    H, _ = cv2.findHomography(srcPoints=src_points, dstPoints=dst_points)
    warp_image = cv2.warpPerspective(src_image, H, (frame_w, frame_h))
    cv2.fillConvexPoly(mask, dst_points, 255)
    cv2.bitwise_and(warp_image, warp_image, frame, mask=mask)

def compute_plot():
    # Crear la figura y los ejes
    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_xlim3d(-1, 1)
    axes.set_ylim3d(-1, 1)
    axes.set_zlim3d(-1, 1)
    line, = axes.plot([], [], [], 'b-')

    return fig, axes

def position_estimation(objPoints, corners, intrinsics, dist_coeffs, axes):
    _, rvec, tvec = cv2.solvePnP(objPoints, corners, intrinsics, dist_coeffs)
                    
    R = cv2.Rodrigues(rvec)[0]
    C = -R.T@ tvec
    C = np.array([int(C[0]), int(C[1]), -int(C[2])]).reshape(3,1)
    A = np.linalg.inv(np.vstack((np.hstack((R, tvec)), np.array([0, 0, 0, 1]))))
    axes.clear() # Limpiar el eje en cada iteración

    return A

def plotImageOnFloor(image, axes):
    """
    Function to plot an image on the 'floor' of the 3D plot.
    """
    # Get the image dimensions
    img_height, img_width, _ = image.shape

    # Generate X and Y coordinates for the floor of the 3D plot
    x = np.linspace(0, img_width, img_width)
    y = np.linspace(0, img_height, img_height)
    X, Y = np.meshgrid(x, y)

    # Place the image on the Z=0 plane (the "floor")
    Z = np.zeros_like(X)

    # Show the image on the 3D floor
    axes.plot_surface(X, Y, Z, rstride=5, cstride=5, facecolors=image / 255.0)

def estimate_distance_from_marker(corners, intrinsics, marker_size_real=8):
    # Extract the focal length (fx, fy) from the camera intrinsics matrix
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    
    # Compute the Euclidean distance between the first two corners to get the marker width in pixels
    # Corners is a 4x2 array, each corner is a (x, y) point
    marker_width_image = np.linalg.norm(corners[0] - corners[1])

    # Average the focal length (fx and fy) in case of slight discrepancies
    f_avg = (fx + fy) / 2.0

    # Apply the formula to compute the distance
    distance = (f_avg * marker_size_real) / marker_width_image
    
    return distance

def plot_image_in_marker(image, intrinsics, dist_coeffs):
    col1, col2 = st.columns(2)
    # Create placeholders for dynamic updates
    with col1:
        st.subheader("Real-Time Image Placing")
        video_placeholder = st.empty()   # Placeholder for the video feed
    with col2:
        st.subheader("Real-Time Position")
        plot_placeholder = st.empty()

    distance_placeholder = st.empty()

    # Se definen el diccionario y los parametros del marcador
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    arucoParams =  cv2.aruco.DetectorParameters()
    objPoints = np.array([[0., 0., 0.], [100., 0., 0.], [100., 100., 0.], [0., 100., 0.]])

    fig, axes = compute_plot()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        opk, frame = cap.read()
        if opk:
            #Paso 2: Deteccion del marcador
            corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=arucoParams)
            if ids is not None:
                total_markers = range(0, ids.size)
                corners = np.concatenate(corners, axis=0)
                #Paso 3: Estimación de posición
                A = position_estimation(objPoints, corners, intrinsics, dist_coeffs, axes)
                #Paso 4: Representación 3D
                plotCamera3D(A, axes)
                # Plot the axis projection
                Pts = np.array([[0, 0, 0, 1],   # origen
                                [150, 0, 0, 1],  # Punto en eje X 
                                [0, 150, 0, 1],  # Punto en eje Y
                                [0, 0, 150, 1]])
                axes.invert_xaxis()
                plot3DAxis(axes, Pts)

                for mark_ids, mark_corners in zip(ids, corners):
                    mark_corners = mark_corners.reshape(4, 2).astype(int)  # Ensure it's in the correct shape
                    image_augmentation(frame, image, mark_corners)

                    distance = estimate_distance_from_marker(mark_corners, intrinsics)
                    
                    # Display the distance in Streamlit
                    distance_placeholder.subheader(f"Estimated Distance to Marker: {distance*0.01:.2f} meters")
                    
                # Importante para que los ejes 3D tengan las mismas proporciones en 
                # matplotlib
                scaling = np.array([getattr(axes, 'get_{}lim'.format(dim))() for dim in 'xyz'])
                axes.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)

                # Actualizar el plot
                fig.canvas.draw()
                fig.canvas.flush_events()

                plot_placeholder.pyplot(fig)

            # Convert the frame (BGR to RGB for displaying in Streamlit)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display the frame in Streamlit
            video_placeholder.image(frame_rgb, channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord('q'):   
            break

    cap.release()
    cv2.destroyAllWindows()