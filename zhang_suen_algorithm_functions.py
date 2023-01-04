import numpy as np
import copy


def compute_A_and_B(square):
    """
    Parameters
    ----------
    square : 2d array
        3x3 matrix of pixels with the pixel (i,j) at the center.

    Returns
    -------
    A : int
        Number of transitions from black --> white among the neighbors.
    B : int
        Number of white pixels among the neighbors.
    """
    
    # Sequence of 8 neighbor nodes (circular)
    sequence = np.array([square[0,1], square[0,2], square[1,2], square[2,2],
                square[2,1], square[2,0], square[1,0], square[0,0],
                square[0,1]], dtype=int) 

    # diff = 0, no transition; diff > 0, black --> white; diff < 0, white --> black
    diff = np.roll(sequence, -1) - sequence 
    
    A = sum(1*(diff > 0)) # Number of transitions from black to white
    B = sum(1 * (sequence[0:8] == 1)) # Number of white pixels among the 8 neighbors
    
    return A, B

def skeletonize(bw_img):
    """
    Parameters
    ----------
    bw_img : 2d array
        Binary image to skeletonize.

    Returns
    -------
    bw_img : 2d array
        Skeletonized binary image.
    """
    
    n_iterations = 0
    # Modify image
    while True:
        
        bw_img, step1_iter, step2_iter = two_steps(bw_img)
        n_iterations += 1
        
        if (step1_iter == 0 and step2_iter == 0):
            break
        
    print(f"Skeletonization is complete. Iterations = {n_iterations}.\n")
    
    return bw_img

def assign_branch_labels(bw_img):
    """
    Parameters
    ----------
    bw_img : 2d array
        Skeletonized binary image.

    Returns
    -------
    branches : 2d array
        Skeletonized binary image where:
            0: black pixel
            1 neighbor: branch endpoint; label = 1
            2 neighbors: regular points; label = 2
            3+ neighbors: branch junctions; label = 3
    """

    xs, ys = np.where(bw_img == 1) # (x,y) indices where the image has white pixels
    rows, cols = np.shape(bw_img)
    branches = np.zeros((rows, cols))
    
    # Iterate through the white pixels
    for q in range(len(xs)):
        
        i, j = xs[q], ys[q]
        
        # If (i,j) is on the edge of the image then we ignore it
        if ((i-1) < 0 or (i+1) >= rows or (j-1) < 0 or (j+1) > cols):
            continue
    
        square = bw_img[i-1:i+2, j-1:j+2]
        
        neighbors = np.array([square[0,1], square[0,2], square[1,2], 
                              square[2,2], square[2,1], square[2,0], 
                              square[1,0], square[0,0]], dtype=int) 
        
        # Assign labels to the pixels based on number of neighbors
        num_white = np.sum(neighbors)
        
        if num_white >= 3: num_white = 3  # Intersection point
        
        if num_white == 1 or num_white == 2:
            
            branches[i,j] = num_white # Regular or endpoint
            
        elif num_white == 3:
            
            boo = check_junction(square) # Junction-checking criteria
            
            if boo == True:
                branches[i,j] = num_white  # Endpoint
            else:
                branches[i,j] = 2 # Regular point
                
    return branches

def corners(square):
    """
    Parameters
    ----------
    square : 2d array
        3x3 matrix of pixels with the pixel (i,j) at the center.

    Returns
    -------
    northeast : int
        Color value (0 or 1) of the (i-1, j+1) pixel.
    northwest : int
        Color value (0 or 1) of the (i-1, j-1) pixel.
    southeast : int
        Color value (0 or 1) of the (i+1, j+1) pixel.
    southwest : int
        Color value (0 or 1) of the (i+1, j-1) pixel.
    """
    
    northeast = square[0,2]
    northwest = square[0,0]
    southeast = square[2,2]
    southwest = square[2,0]
    
    return northeast, northwest, southeast, southwest

def merge(list1, list2):
    """
    Parameters
    ----------
    list1 : list
        First list.
    list2 : list
        Second list.

    Returns
    -------
    merged_list : list
        Merged list of list1 and list2 where the elements are tuples.
    """
      
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list 

def find_branches_from_endpoints(B_set, bw_img, branches):
    """
    Parameters
    ----------
    B_set : list
        Set of branches for storing new branches.
    bw_img : 2d array
        Skeletonized binary image.
    branches : 2d array
        Skeletonized binary image where:
            0: black pixel
            1 neighbor: branch endpoint; label = 1
            2 neighbors: regular points; label = 2
            3+ neighbors: branch junctions; label = 3

    Returns
    -------
    B_set : list
        Updated set of branches.
    """

    rows, cols = np.shape(bw_img)
    endpoints = np.where(branches == 1)
    N_endpoints = len(endpoints[0])
    
    for q in range(N_endpoints):
        
        endpoint = (endpoints[0][q], endpoints[1][q])
        
        B = list() # Empty branch xs
        
        i, j = endpoint[0], endpoint[1] # CURRENT POINT
    
        square = branches[i-1:i+2, j-1:j+2]
        north, south, east, west = nsew(square)
        
        if north == 2:
            new = (i-1, j)
        elif south == 2: 
            new = (i+1, j)
        elif east == 2:
            new = (i, j-1)
        elif west == 2:
            new = (i, j+1)
        else:
                northeast, northwest, southeast, southwest = corners(square)
                
                if northeast ==2:
                    new = (i-1, j+1)
                elif northwest ==2:
                    new = (i-1, j-1)
                elif southeast ==2:
                    new = (i+1, j+1)
                elif southwest ==2:
                    new = (i+1, j-1)
        
        # ADD NEW POINT
        B.append(new)
        
        prev = (i, j) # CURRENT --> PREVIOUS
        i, j = new   # NEW --> CURRENT
        
        while branches[new[0], new[1]] != 3:
            
            square = branches[i-1:i+2, j-1:j+2]
            north, south, east, west = nsew(square)
            
            # DO NOT WANT TO STORE PREV POINT
            if north !=0 and ((i-1, j) != prev) and ((i-1,j) not in B):
                new = (i-1, j)
            elif south !=0 and ((i+1, j) != prev) and ((i+1,j) not in B): 
                new = (i+1, j)
            elif east !=0 and ((i,j+1) != prev) and ((i,j+1) not in B):
                new = (i, j+1)
            elif west !=0 and ((i,j-1) != prev) and ((i,j-1) not in B):
                new = (i, j-1)
            else:
                
                northeast, northwest, southeast, southwest = corners(square)
                
                if northeast !=0 and ((i-1, j+1) != prev) and ((i-1,j+1) not in B):
                    new = (i-1, j+1)
                elif northwest !=0 and ((i-1, j-1) != prev) and ((i-1,j-1) not in B):
                    new = (i-1, j-1)
                elif southeast !=0 and ((i+1, j+1) != prev) and ((i+1,j+1) not in B):
                    new = (i+1, j+1)
                elif southwest !=0 and ((i+1, j-1) != prev) and  ((i+1,j-1) not in B):
                    new = (i+1, j-1)
    
            B.append(new)
            prev = (i, j) # CURRENT --> PREVIOUS
            i, j = new  # NEW --> CURRENT
    
        B.insert(0, endpoint)
        B_set.append(B)
    
    return B_set

def nsew(square):
    """
    Parameters
    ----------
    square : 2d array
        3x3 matrix of pixels with the pixel (i,j) at the center.

    Returns
    -------
    north : int
        Color value (0 or 1) of the (i-1, j) pixel.
    south : int
        Color value (0 or 1) of the (i+1, j) pixel.
    east : int
        Color value (0 or 1) of the (i, j+1) pixel.
    west : int
        Color value (0 or 1) of the (i, j-1) pixel.
    """
    
    north = square[0,1]
    south = square[2,1]
    east = square[1,2]
    west = square[1,0]
    
    return north, south, east, west

def step1_condition(square):
    """
    Parameters
    ----------
    square : 2d array
        3x3 matrix of pixels with the pixel (i,j) at the center.

    Returns
    -------
    bool
        T/F depending on whether the step 1 condition is met.
    """
    north, south, east, west = nsew(square)
    
    if ((north * east * south) == 0 and (east * south * west) == 0):
        return True
    else:
        return False
    
def step2_condition(square):
    """
    Parameters
    ----------
    square : 2d array
        3x3 matrix of pixels with the pixel (i,j) at the center.

    Returns
    -------
    bool
        T/F depending on whether the step 2 condition is met.
    """
    north, south, east, west = nsew(square)
    
    if ((north * east * west) == 0 and (north * south * west) == 0):
        return True
    else:
        return False

def execute_step(bw_img, step_condition):
    """
    Parameters
    ----------
    bw_img : 2d array
        Binary imge.
    step_condition : function
        Input either `step1_condition` for step 1 or `step2_condition` for step 2.

    Returns
    -------
    bw_img : 2d array
        Thinned binary image.
    step_iter : int
        Number of pixels removed in step 1 or step 2.
    """
    
    rows, cols = np.shape(bw_img) # image dimensions
    step_iter = 0
    xy_mod = list()
    xs, ys = np.where(bw_img == 1) # (x,y) indices where the image has white pixels
    for q in range(len(xs)):
        
        i, j = xs[q], ys[q]

        # If (i,j) is on the edge of the image then we ignore it
        if ((i-1) < 0 or (i+1) >= rows or (j-1) < 0 or (j+1) > cols):
            continue

        square = bw_img[i-1:i+2, j-1:j+2]
        A, B = compute_A_and_B(square)
        
        if ( A == 1 and 
            (2 <= B <= 6) and
            step_condition(square) == True):
            
            xy_mod.append([i,j])
            step_iter += 1

    # Modify the pixels
    for i,j in xy_mod: bw_img[i,j] = 0
    
    return bw_img, step_iter

def two_steps(bw_img):
    """
    Parameters
    ----------
    bw_img : 2d array
        Binary image.

    Returns
    -------
    bw_img : 2d array
        Thinned binary image.
    step1_iter : int
        Number of pixels removed in step 1.
    step2_iter : int
        Number of pixels removed in step 2.
    """
    
    bw_img, step1_iter = execute_step(bw_img, step1_condition) # STEP 1
    bw_img, step2_iter = execute_step(bw_img, step2_condition) # STEP 2
    
    return bw_img, step1_iter, step2_iter
    
def check_junction(square):
    """
    Parameters
    ----------
    square : 2d array
        3x3 matrix of pixels with the pixel (i,j) at the center.

    Returns
    -------
    Boolean
        True = (i,j) is an intersection, False = (i,j) is a regular point.
    """
    
    north, south, east, west = nsew(square)
    corners = np.array([square[0,0], square[0,2], 
                        square[2,0], square[2,2]])
    
    nsew_sum = int(north + south + east + west)
    corn_sum = int(np.sum(corners))
    
    square = square.astype(int)
    
    mat_1 = np.array([[1, 0, 1],
                      [0, 1, 0],
                      [0, 1, 0]])
    
    mat_2 = np.array([[1, 0, 1],
                      [0, 1, 1],
                      [0, 1, 0]])
    
    mat_3 = np.array([[1, 0, 1],
                      [1, 1, 0],
                      [0, 1, 0]])
    
    mat_4 = np.array([[0, 1, 0],
                      [1, 1, 0],
                      [0, 0, 1]])
    
    mat_5 = np.array([[0, 1, 0],
                      [0, 1, 1],
                      [1, 0, 0]])
    
    mat_6 = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [1, 0, 0]])
    
    mat_7 = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 0, 1]])
    
    mat_8 = np.array([[0, 1, 0],
                      [0, 1, 1],
                      [1, 1, 0]])
    
    mat_9 = np.array([[1, 1, 0],
                      [0, 1, 1],
                      [0, 1, 0]])
    
    
    if (nsew_sum >= 3) and (corn_sum == 0):
        return True
    elif (corn_sum >= 3) and (nsew_sum == 0) :
        return True
    else:
        
        satisfies_mat = False
        
        for i in range(3):
            
            if np.array_equal(square, mat_1) == True:
                satisfies_mat = True
                break
            elif np.array_equal(square, mat_2) == True:
                satisfies_mat = True
                break
            elif np.array_equal(square, mat_3) == True:
                satisfies_mat = True
                break
            elif np.array_equal(square, mat_4) == True:
                satisfies_mat = True
                break
            elif np.array_equal(square, mat_5) == True:
                satisfies_mat = True
                break
            elif np.array_equal(square, mat_6) == True:
                satisfies_mat = True
                break
            elif np.array_equal(square, mat_7) == True:
                satisfies_mat = True
                break
            elif np.array_equal(square, mat_8) == True:
                satisfies_mat = True
                break
            elif np.array_equal(square, mat_9) == True:
                satisfies_mat = True
                break
            else:
                mat_1 = np.rot90(mat_1)
                mat_2 = np.rot90(mat_2)
                mat_3 = np.rot90(mat_3)
                mat_4 = np.rot90(mat_4)
                mat_5 = np.rot90(mat_5)
                mat_6 = np.rot90(mat_6)
                mat_7 = np.rot90(mat_7)
                mat_8 = np.rot90(mat_8)
                mat_9 = np.rot90(mat_9)

        return satisfies_mat
    
def investigate_new_endpoints(branches, B_set):
    """
    Parameters
    ----------
    branches : 2d array
        Skeletonized binary image where:
            0: black pixel
            1 neighbor: branch endpoint; label = 1
            2 neighbors: regular points; label = 2
            3+ neighbors: branch junctions; label = 3
            
    B_set : list
        Stores the segmented skeleton branches; is the set of branches B.

    Returns
    -------
    branches_new : 2d array
        Updated `branches` variable.
    """
    # Now, we want to subtract the currrent branches (except for intersection points)
    for q in range(len(B_set)):
        
        B = B_set[q]
        
        for i in range(len(B) - 1): # all except intersection point
        
            x_val, y_val = B[i][0], B[i][1]
            branches[x_val, y_val] = 0
        
    # Now, we want all the points to be binary (white = 1, black = 0)
    xs, ys = np.where(branches != 0) # (x,y) indices where the image has white pixels
    branches_new = copy.deepcopy(branches)
    for i in range(len(xs)):
        
        x_val, y_val = xs[i], ys[i]
        branches_new[x_val, y_val] = 1
    
    # Now, we run the labeling algorithm again
    # Assign branch labels
    branches_new = assign_branch_labels(branches_new)

    reg_xs, reg_ys = np.where(branches_new == 2)
    for i in range(len(reg_xs)):
        
        x_val, y_val = xs[i], ys[i]
        
        if branches[x_val, y_val] == 3 and branches_new[x_val, y_val] != 1:
            branches_new[x_val, y_val] = 3
            
    return branches_new

def find_all_other_branches(B_set, branches, bw_img):
    """
    Parameters
    ----------
    B_set : list
        Stores the segmented skeleton branches; is the set of branches B.
    branches : 2d array
        Skeletonized binary image where:
            0: black pixel
            1 neighbor: branch endpoint; label = 1
            2 neighbors: regular points; label = 2
            3+ neighbors: branch junctions; label = 3
    bw_img : 2d array
        Skeletonized binary image.

    Returns
    -------
    B_set : list
        Updated `B_set`.
    """

    # Add all other branches that do not form a loop
    while True:
        
        # Update branches
        branches = investigate_new_endpoints(branches, B_set)
    
        # We can't fit too short of segments
        xs, ys = np.where(branches != 0)
        if len(xs) <= 7:
            break
        
        # Update the set
        B_set = find_branches_from_endpoints(B_set, bw_img, branches)
        
        # There are no more "endpoints"
        xs, ys = np.where(branches == 1)
        if len(xs) == 0:
            break
    
    # CASE WHEN IT IS A LOOP
    xs, ys = np.where(branches == 3)
    if len(xs) == 2:
        
        x1, y1 = xs[0], ys[0]
        x2, y2 = xs[1], ys[1]
        branches[x1, y1] = 1 # Make one of them an endpoint
        
        # Add 1st branch
        B_set = find_branches_from_endpoints(B_set, bw_img, branches)
        
        # Modify branches
        branches = investigate_new_endpoints(branches, B_set)
        branches[x2, y2] = 3 # Make one of them an intersection point
        
        # Add 2nd branch
        B_set = find_branches_from_endpoints(B_set, bw_img, branches)
        branches = investigate_new_endpoints(branches, B_set)
        
    return B_set

def split_all_branches(bw_img, branches):
    """
    Parameters
    ----------
    bw_img : 2d array
        Skeletonized binary image.
    branches : 2d array
        Skeletonized binary image where:
            0: black pixel
            1 neighbor: branch endpoint; label = 1
            2 neighbors: regular points; label = 2
            3+ neighbors: branch junctions; label = 3

    Returns
    -------
    B_set : list
        Compelete set of branches B that can be fitted to B-splines.
    """
    
    B_set = list()
    B_set = find_branches_from_endpoints(B_set, bw_img, branches)
    B_set = find_all_other_branches(B_set, branches, bw_img)
    
    return B_set