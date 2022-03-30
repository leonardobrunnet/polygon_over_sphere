#!/usr/bin/env python
# coding: utf-8

# In[437]:


import torch
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[580]:


def convert_cartesian_tensor_to_spherical(X):
    Rs = torch.norm(X,dim=1)
    theta = torch.atan2(X[:,1],X[:,0])
    phi = torch.asin(X[:,2]/Rs[:])
    return Rs,theta,phi

def convert_spherical_tensor_to_cartesian(Rs,theta,phi):
    x=torch.zeros(len(Rs),3,device=device)
    x[:,0] = Rs[:]*torch.cos(phi[:])*torch.cos(theta[:])
    x[:,1] = Rs[:]*torch.cos(phi[:])*torch.sin(theta[:])
    x[:,2] = Rs[:]*torch.sin(phi[:])
    return x

def convert_cartesian_to_spherical(X):
    Rs = torch.norm(X)
    theta = torch.atan2(X[1],X[0])
    phi = torch.asin(X[2]/Rs)
    #Rs_theta_phi_cm=torch.stack((Rs,theta,phi),0)
    return  Rs,theta,phi

def convert_cartesian(Rs,theta,phi):

    x = torch.zeros(3,device=device)
    x[0] = Rs*torch.cos(phi)*torch.cos(theta)
    x[1] = Rs*torch.cos(phi)*torch.sin(theta)
    x[2] = Rs*torch.sin(phi)
    return x

def generate_polar_circle(r,Rs,num_part):
    rho = torch.arcsin(r/Rs)
    phi = np.pi/2-rho
    X = torch.zeros(num_part,3,device=device)
    dtheta = torch.tensor(2*np.pi/num_part,device=device)
    for i in range(num_part):
        X[i,:]=convert_cartesian(Rs,i*dtheta,phi)
    return X

def generate_circle_at_theta_phi(num_part,r,Rs,theta,phi):
    delta = phi + torch.tensor(np.pi)/2
    Z = generate_polar_circle(r,Rs,num_part)
    Z = rotate_around_y(Z,delta)
    Z = rotate_around_z(Z,theta)
    return Z

def rotate_around_y(X,phi):
    delta = torch.tensor(np.pi)/2 - phi
    Y = torch.zeros_like(X)
    Y[:,0] = X[:,0]*torch.cos(delta) + X[:,2]*torch.sin(delta)
    Y[:,1] = X[:,1]
    Y[:,2] = X[:,2]*torch.cos(delta) - X[:,0]*torch.sin(delta)    
#     Y = torch.zeros_like(X)
#     Y[:,0] = -X[:,0]*np.sin(phi) + X[:,2]*np.cos(phi)
#     Y[:,1] = X[:,1]
#     Y[:,2] = -X[:,2]*np.sin(phi) - X[:,0]*np.cos(phi)
    return Y
    
def rotate_around_z(X,theta):
    Y = torch.zeros_like(X)
    Y[:,0] = X[:,0]*torch.cos(theta) - X[:,1]*torch.sin(theta)
    Y[:,1] = X[:,1]*torch.cos(theta) + X[:,0]*torch.sin(theta)
    Y[:,2] = X[:,2]
    return Y



def rotate_from_zenite_to_theta_phi(Z,theta,phi):
    Z = rotate_around_y(Z,phi)
    Z = rotate_around_z(Z,theta)
    return Z
    
def func_d_lamb(r,Rs,delta):
    L = Rs * torch.sin(delta)
    d_lamb = torch.arcsin(2.1*r/L)
    return d_lamb
    
def generate_image_points(X):
    xs = X[:,0]
    ys = X[:,1]
    zs = X[:,2]
    return xs,ys,zs



# In[518]:


def make_3d_image(poly_list):
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
    for i in poly_list:
        xs,ys,zs=generate_image_points(i.X)
        xs,ys,zs=xs.to("cpu"),ys.to("cpu"),zs.to("cpu")
        ax.scatter(xs,ys,zs)
    ax.view_init(90,0)
#plt.show()


# In[519]:


def make_3d_image_cm(poly_list):
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
    for i in poly_list:
        xs,ys,zs=i.cm.cpu()
        ax.scatter(xs,ys,zs)
    #ax.view_init(np.pi/4,np.pi/4)
#plt.show()


# In[520]:


def neighbors(poly_list,r):
    lp=len(poly_list)
    X_cm=torch.zeros(lp,3,device=device)
    for i in poly_list:
        X_cm[i.ident]=i.cm
    D=dist_mat_square(X_cm)
    D += torch.eye(lp,device=device)*10**4
    one = torch.ones(lp,device=device)
    zero = torch.zeros(lp,device=device)
    D = torch.where(D<3*r,one,zero)
    z=torch.tensor(np.arange(1,lp+1),device=device)
    for i in poly_list:
        y = list(map(int,((D[i.ident]*z)).tolist()))
        y = list(filter(lambda num: num != 0, y[i.ident:]))
        y = list(i-1 for i in y)
        i.neighbor_list=y
    return
    


# In[521]:


def dist_mat_square(X):
    return torch.sqrt(torch.sum((X[:,None,:]-X[None,:,:])**2,axis=2))


# In[522]:


def distmat_square_inter_polygons(X,Y):
    return torch.sqrt(torch.sum((X[:,None,:]-Y[None,:,:])**2,axis=2))


# In[523]:


def force_mod(R,zero_tensor,Req,R0,Frep,Fadh):
    frep  = -Frep*(1-Req/R)
    frep  = torch.where(R<Req,frep,zero_tensor)
    fadh  = -Fadh*(1-Req/R)
    fadh  = torch.where(R>Req,fadh,zero_tensor)
    fadh  = torch.where(R<R0,fadh,zero_tensor)
    force = fadh+frep
    return  force


# In[524]:


def force_mod_internal_polygon(R,Req,Frep,zero_tensor):
    frep = -Frep*(1-Req/R)
    frep = torch.where(R<Req,frep,zero_tensor)
    return frep


# In[525]:


def force_field_inter_polygons(X,Y,R,zero_tensor,Req,R0,Frep,Fadh):
    force = force_mod(R,zero_tensor,Req,R0,Frep,Fadh)
    FF_target_polygon = torch.sum(force[:,:,None]*((X[:,None,:]-Y[None,:,:])),axis=1)
    FF_reaction = -torch.sum(force[:,:,None]*((X[:,None,:]-Y[None,:,:])),axis=0)
    return FF_target_polygon,FF_reaction


# In[526]:


def force_field_intra_polygon(X,R,zero_tensor,Req,Frep):
    force = force_mod_internal_polygon(R,Req,Frep,zero_tensor)
    FF = torch.sum(force[:,:,None]*((X[:None,:]-X[None,:,:])),axis=1)
    return FF


# In[646]:


class polygon:
    Req = 1.
    ks = 10
    R0 = 1.2
    Frep = 20.
    Fadh = 8.
    mu = 1.
    
    def __init__(self,ident,X):
        self.X = X.clone().detach().to(device)
        self.ident = ident
        self.num_part = len(X)
        self.Force = torch.zeros(self.num_part,3,device=device)
        self.neighbor_list = []
        self.maxforce = 0
        self.kb = kb*self.num_part
        
    def perimeter_and_self_repulsion_forces(self):
        Xp = torch.cat((self.X[:,:],self.X[None,0,:]),0)#adding first particle postion to the end
        self.Xd = Xp[1:,:]-self.X   #right distance between particles
        self.Xe = -torch.cat((self.Xd[None,-1,:],self.Xd[:-1,:]),0) #left distance between particles
        R = torch.sqrt(torch.sum(self.Xd**2,axis=1)) 
        fnormd = self.ks*(R-self.Req)
        fnorme = torch.cat((fnormd[None,-1],fnormd[:-1]))
        f_vec_d = self.Xd * torch.div(fnormd,torch.norm(self.Xd,dim=1))[:,None]
        f_vec_e = self.Xe * torch.div(fnorme,torch.norm(self.Xe,dim=1))[:,None]  
        self.Force = f_vec_d + f_vec_e
        #Repulsion if not first neighbors when too close
        R = dist_mat_square(self.X)
        z = torch.ones(self.num_part,device=device)
        tridiag = (torch.diag(z[:-1],1)+torch.diag(z)+torch.diag(z[:-1],-1))*1000 # large tridiagonal matrix
        tridiag = tridiag.to(device)
        R += tridiag #summing a large tridiagonal to avoid auto zero distance and already calculated neighbors
        zero_tensor = torch.zeros(self.num_part,self.num_part,device=device)
        FF = force_field_intra_polygon(self.X,R,zero_tensor,self.Req,self.Frep)
        self.Force += FF
        
        return
    
    def bending_force(self):
        X_perp_d=torch.cross(self.Xd,self.X,axis=1) #vector perpendicular to membrane right
        X_perp_e=torch.cross(self.X,self.Xe,axis=1) #vector perpendicular to membrane left
        X_perp_d=torch.div(X_perp_d,torch.norm(X_perp_d,dim=1)[:,None]) #normalizing
        X_perp_e=torch.div(X_perp_e,torch.norm(X_perp_e,dim=1)[:,None]) #normalizing        
        self.angle = torch.arccos(torch.sum(X_perp_d*X_perp_e,axis=1)) #scalar product of unitary vectors
        X_perp_sum = X_perp_e+X_perp_d  #central bending vector
        X_perp_sum = torch.div(X_perp_sum,torch.norm(X_perp_sum,dim=1)[:,None]) #normalizing
        self.Force += -self.kb*self.angle[:,None]*X_perp_sum    #bending force on focus particle
        index_d=list([i for i in range(1,len(self.angle))])+[0]     
        index_e=list([i for i in range(-1,len(self.angle)-1)])       
        #forces on focus site and first neighbor should sum zero
        self.Force[index_d] +=  self.kb*self.angle[:,None]*X_perp_d/(2.*torch.cos(self.angle[:,None]/2.)) 
        self.Force[index_e] +=  self.kb*self.angle[:,None]*X_perp_e/(2.*torch.cos(self.angle[:,None]/2.))
        self.maxforce=max(torch.norm(self.Force,dim=1)).item()
        return 
    
    def move_cell_particles(self):
        dx = self.mu*self.Force*dt

        self.X += dx
        #back to the surface
        R,theta,phi = convert_cartesian_tensor_to_spherical(self.X)
        R=torch.ones(self.num_part,device=device)*Rs0
        self.X=convert_spherical_tensor_to_cartesian(R,theta,phi)


        return
    
    def zero_forces(self):
        self.Force = torch.zeros(self.num_part,3,device=device)
        return
    
    def center_of_mass(self):
        self.cm = torch.sum(self.X,0)/len(self.X)
        return
    
    def inter_polygon_forces(self):
        for i in self.neighbor_list :
            R = distmat_square_inter_polygons(self.X,poly_list[i].X)
            zero_tensor=torch.zeros(self.num_part,self.num_part,device=device)
            myForce,reactionForce=force_field_inter_polygons(self.X,poly_list[i].X,R,zero_tensor,self.Req,self.R0,self.Frep,self.Fadh)
                                                          #force_field_inter_polygons(X,Y,R,zero_tensor,Req,R0,Frep,Fadh         

            self.Force+=myForce
            poly_list[i].Force+=reactionForce       
        return

    


# In[647]:


def division(poly_list,r,division_axis):
    pi = torch.tensor(np.pi)
    number_of_cells = len(poly_list)
    mother_cell = np.random.randint(number_of_cells)
    num_part = len(poly_list[mother_cell].X)
    Xcm_mother = poly_list[mother_cell].cm
    R,theta,phi = convert_cartesian_to_spherical(Xcm_mother)
    dphi=torch.arcsin(1.1*r/(2*R))
    X_mother=generate_circle_at_theta_phi(num_part,r/2.2,R,division_axis*pi/2,dphi) #at zenit
    X_mother=rotate_from_zenite_to_theta_phi(X_mother,theta,phi)
    poly_list[mother_cell].X = X_mother
    X_daughter=generate_circle_at_theta_phi(num_part,r/2.2,R,division_axis*pi/2,-dphi) #at zenit
    X_daughter = rotate_from_zenite_to_theta_phi(X_daughter,theta,phi)
    new_index=len(poly_list)
    poly_list.append(polygon(new_index,X_daughter))
    return


# In[648]:


def init():
    global t, kb, Rs0,dt
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device="cpu"
    print("device=",device)
    dt = 0.004
    kb = 3.
    num_part = 30        #number of particles per polygon
    Req = 1        #equilibrium distance between polygon particles
    r=torch.tensor(num_part*Req/(2*np.pi)) #polygon "radius"
    Rs0 = 10*r       #sphere radius
    # Determine maximum number of polygons
    max_poly = int(4*Rs0**2/r**2)
    #set max number of polygons
    poly = 1
    if poly > max_poly : 
        print("Too much cells on the sphere, exiting...")
        exit()
    #Generate polygons on the sphere
    poly_list = []
    X=generate_polar_circle(r,Rs0,num_part) #circle centered at zenith
    X = X.to(device)
    poly_list.append(polygon(0,X)) #first cell
    return poly_list,r
        


# In[ ]:
                     

#main program
poly_list,r= init()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#make_3d_image(poly_list)
dt0=dt
t,tf,division_time,finish_division= 0.,40., 10.0,35.
next_division=division_time
division_axis = 0
intt=0
time0=time.time()
while t < tf:
    t+=dt
        
    list(map(lambda i:i.zero_forces(),poly_list))
    list(map(lambda i:i.perimeter_and_self_repulsion_forces(),poly_list))
    if kb > 0: list(map(lambda i:i.bending_force(),poly_list))
    list(map(lambda i:i.center_of_mass(),poly_list))
    neighbors(poly_list,r)
    list(map(lambda i:i.inter_polygon_forces(),poly_list))
    maxforce_list=[]
    for i in poly_list:
        maxforce_list.append(i.maxforce)
    #print(maxforce_list)
    fmax=max(maxforce_list)
    #print(fmax)
    dt=min(0.004/fmax,0.01)
    if int(t)>intt:
        print(t,dt)
        intt=int(t)
    if t > finish_division and t < finish_division+dt:
        print("Finished dividing. Reseting kb to lower value")
        kb=kb/10
    list(map(lambda i:i.move_cell_particles(),poly_list))
    if t>next_division and t<finish_division:
        division_axis=np.mod(division_axis,2) #division once in x another in y
        division(poly_list,r,division_axis)
        division_axis+=1
        next_division+=division_time
#     plt.show()
print(time.time()-time0)
list(map(lambda i:i.center_of_mass(),poly_list))
#make_3d_image_cm(poly_list)
make_3d_image(poly_list)
    #exit()
plt.show()






