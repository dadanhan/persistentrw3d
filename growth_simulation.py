import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def generate_newx(position,speed,dt,maxdangle=None,oldtheta=None,oldphi=None):
    '''this function generates a new x position given current position(3d np.array)'''
    #generate a random increment to step forward
    if maxdangle is None or oldtheta is None or oldphi is None:
        newphi = random.uniform(0,np.pi)
        newtheta = random.uniform(0,2*np.pi)
    else:
        dtheta = random.uniform(-maxdangle,maxdangle)
        dphi = random.uniform(-maxdangle,maxdangle)
        newtheta = oldtheta+dtheta
        newphi = oldphi+dphi
    rand_v = speed*np.array([np.cos(newtheta)*np.sin(newphi),np.sin(newtheta)*np.sin(newphi),np.cos(newphi)],dtype=float)
    dx = rand_v*dt
    newx = position+dx
    return newx, newphi, newtheta

def PRW_branching3d(tend,v,x,dt,boundradius,dangle,pbranch,touch_dist,inertial=False):
    #initialize simulation parameters 
    t = 0
    numnrn = len(x)
    #define list of nodes to grow, format of each entry is (neuron id, node id)
    nodes2grow = []
    #initialize data array to store simulation steps
    for i in range(0,numnrn):
        #format of data columns is neuron id, node id, parent id, x, y, z, phi, theta, time
        if i == 0:
            data = np.array([[i,0,-1,x[i][0],x[i][1],x[i][2],random.uniform(0,np.pi),random.uniform(0,2*np.pi),t]],dtype=float)
        else:
            data = np.concatenate(( data, np.array([[i,0,-1,x[i][0],x[i][1],x[i][2],random.uniform(0,np.pi),random.uniform(0,2*np.pi),t]],dtype=float) ))
        nodes2grow.append((i,0))
    #define boundary distance to stop
    threshdistsq = np.power(touch_dist,2)
    threshboundr = np.power(boundradius - touch_dist,2)
    
    while t < tend:
        newnodes2grow = []
        for items in nodes2grow:
            #get index in data where the node to grow is located
            node2grow_index = np.where((data[:,0] == items[0]) & (data[:,1] == items[1]))[0][0]
            #get the positions and angles
            tempdata = data[node2grow_index]
            tempnrnid = tempdata[0]
            tempx = tempdata[3:6]
            tempphi = tempdata[6]
            temptheta = tempdata[7]
            #check if node is within threshold distance of the boundary
            dist_from_origin = np.sum(np.power(tempx,2))
            if dist_from_origin < threshboundr: #if your current node position is within the bounds of the simulation, continue
                #check if node is within threshold distance of another neuron (and not itself)
                dist_from_nrns = np.sum(np.power(tempx-data[data[:,0] != items[0]][:,3:6],2),axis=1)
                if (dist_from_nrns > threshdistsq).all(): #if your current node position is not within a threshold distance of another neuron
                    #grow next node
                    #generate new position from current position
                    if inertial==True:
                        newx,newphi,newtheta = generate_newx(tempx,v,dt,dangle,temptheta,tempphi)
                    else:
                        newx,newphi,newtheta = generate_newx(tempx,v,dt)
                    newnid = np.amax(data[data[:,0]==items[0],1]+1)
                    newpid = tempdata[1]
                    #add new node to data array
                    data = np.concatenate(( data, np.array([[tempnrnid,newnid,newpid,newx[0],newx[1],newx[2],newphi,newtheta,t+dt]],dtype=float) ))
                    newnodes2grow.append((tempnrnid,newnid))
                    #flip a coin and see if you branch and then repeat the process
                    U = np.random.uniform(0,1)
                    if U < pbranch:
                        if inertial==True:
                            newx,newphi,newtheta = generate_newx(tempx,v,dt,dangle,temptheta,tempphi)
                        else:
                            newx,newphi,newtheta = generate_newx(tempx,v,dt)
                        newnid = np.amax(data[data[:,0]==items[0],1]+1)
                        newpid = tempdata[1]
                        data = np.concatenate(( data, np.array([[tempnrnid,newnid,newpid,newx[0],newx[1],newx[2],newphi,newtheta,t+dt]],dtype=float) ))
                        newnodes2grow.append((tempnrnid,newnid))
        #swap nodes2grow with newnodes2grow list for next time step iteration
        nodes2grow = newnodes2grow
        #update simulation time
        t += dt
    return data

def plot_data(data,boundr):
    print(data)
    nrnids = np.unique(data[:,0])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in nrnids:
        tempdata = data[data[:,0]==i]
        if tempdata is None:
            pass
        else:
            # Extracting node id, parent id, x, y, z coordinates
            node_ids = np.array(tempdata[:,1],dtype=int)
            parent_ids = np.array(tempdata[:,2],dtype=int)
            x = np.array(tempdata[:,3],dtype=float)
            y = np.array(tempdata[:,4],dtype=float)
            z = np.array(tempdata[:,5],dtype=float)

            # Find the soma node (with parent id -1)
            soma_index = np.where(parent_ids == -1)[0]
            ax.plot(x[soma_index], y[soma_index], z[soma_index],'.',markersize=30, color='cyan', alpha=0.7)

            # Creating lines between nodes and their parent nodes using Line3DCollection
            lines = []
            for i in range(0,len(node_ids)):

                if parent_ids[i] != -1:  # Ignore if parent id is -1 (root node)
                    pdex = np.where(node_ids == parent_ids[i])[0][0]
                    # print(pdex)
                    line = [(x[i], y[i], z[i]), (x[pdex], y[pdex], z[pdex])]
                    lines.append(line)

            lc = Line3DCollection(segments=lines, linewidths=0.5, color='blue')
            ax.add_collection(lc)

            #find all nodes with no children (the ends) or two children (the branch points) and mark them separately
            for i in range(0,len(node_ids)):
                num_children = len(np.where(parent_ids == node_ids[i])[0])
                if num_children == 0: #found end
                    ax.plot(x[i],y[i],z[i],'.',markersize=10,color='cyan',alpha=0.7)
                if num_children == 2: #found branch point
                    ax.plot(x[i],y[i],z[i],'.',markersize=10,color='green',alpha=0.7)
    ax.set_xlim(-boundr,boundr)
    ax.set_ylim(-boundr,boundr)
    ax.set_zlim(-boundr,boundr)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    endtime = 100
    dt = 0.1
    pbranch = 0.02*dt
    velocity = 1
    maxdangle = np.pi/8
    boundr = 10
    touchdist = 0.1

    num_nrns = 10
    pos = np.zeros((num_nrns,3),dtype=float)
    for i in range(0,num_nrns):
        phi = np.random.uniform(0,np.pi)
        theta = np.random.uniform(0,2*np.pi)
        r = np.random.uniform(0,boundr)
        pos[i,0] = r*np.cos(theta)*np.sin(phi)
        pos[i,1] = r*np.sin(theta)*np.sin(phi)
        pos[i,2] = r*np.cos(phi)

    
    data = PRW_branching3d(endtime,velocity,pos,dt,boundr,maxdangle,pbranch,touchdist)

    plot_data(data,boundr)

    data = PRW_branching3d(endtime,velocity,pos,dt,boundr,maxdangle,pbranch,touchdist,inertial=True)

    plot_data(data,boundr)
    
