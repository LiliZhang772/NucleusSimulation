#!/usr/bin/env python
# coding: utf-8

import numpy as np



image_size = 120 # px (The image will be square 100 x 100 x 100)
pixel_size = 0.1 # um
boundary = image_size * pixel_size 
radius =  3; # um
dwell_time = 0.001 # s
psf_width = 0.3 # um (Width of the point spread function in focus)
psf_height = 1.5 # 
diff_const = 1 # um^2/s (diffusion coefficient of mobile particles)
step_time = 0.001 # s 
B = 1e4 # Brightness, Hz

Nparticles = 5000


# The geometric parameters of the detection volume.
Ea  = 0.3 # um
Eb  = 0.3
Ec  = 2.1 

z_slice = [59, 89] # the z slice that is to be imaged.


# genreate initial positions of particles, which are in the lower half of the sphere.

start_pos = np.zeros((Nparticles,3))
for n in range(Nparticles):
    temp = start_pos[n,:]
    while temp[0]**2 + temp[1]**2 + temp[2]**2 == 0:
        x = np.random.rand(3) * 12
        if ((x[0] - 6)**2 + (x[1] - 6)**2 + (x[2] - 6)**2) < radius**2 :
            start_pos[n,:] = x

center_pos = [6, 6, 6]


# In[ ]:


# ### Calculating the pixel intensity
# The pixel intensity is dependent on the distance from the optical axis

def GaussianBeam( start_pos, beam_pos, psf_width, psf_height):
    if start_pos.shape[0] == 2:
        GB = step_time*np.exp(- 2* ((start_pos - beam_pos)**2).sum()/ psf_width**2) # the B will be added later
    else:
        GB = step_time*np.exp(- 2* ((start_pos[0:2] - beam_pos[0:2])**2).sum()/ psf_width**2) * np.exp(-2*((start_pos[2]-beam_pos[2])**2/psf_height**2))
        
    return GB





# Calculate the image array when the particles are stationary.

image_array = np.zeros((image_size,image_size,len(z_slice)))
image_array_stationary = np.zeros((image_size,image_size,len(z_slice)))

for n in range(Nparticles):
    particle_pos = start_pos[n]   
    for k in z_slice : # z
        for j in range(image_array.shape[1]): # x
            for i in range(image_array.shape[0]): # y
                
                    kk = round(k/30-2) ;
                    beam_pos = np.array([i,j,k]) * pixel_size 
                    
                    image_array[i,j,kk] += B* GaussianBeam(particle_pos,beam_pos,psf_width,psf_height)

image_array_stationary = np.array(image_array)


# In[ ]:
# Setting up parameters for the movement of all the particles for all the steps in the simulation.

steps = image_size*image_size*len(z_slice)

pre_pos = np.zeros((steps+1,3,Nparticles))
pre_pos[0,:,:] = np.transpose(start_pos)
depth = np.zeros((steps,Nparticles))

track = np.random.normal(loc=0,scale=np.sqrt(2*diff_const*step_time),size=(steps,3,Nparticles))

loca = np.zeros((steps,3,Nparticles))
CB = np.ones((steps,Nparticles))

# In[ ]:
# Movement of each particle during each timestep.
    
for n in range(Nparticles):
    for i in range(steps):
    
        depth[i,n] = np.sqrt(((pre_pos[i,:,n] - center_pos)**2).sum())
        forwd = np.sqrt(((pre_pos[i,:,n] + track[i,:,n] - center_pos)**2).sum())

        # In this case, only diffusion inside the sphere is allowed.
        if forwd <= radius:
                loca[i,:,n] = pre_pos[i,:,n] + track[i,:,n]

        else:
               loca[i,:,n] = pre_pos[i,:,n] 
        
        # Set the molecular brightness to 0 if the particle is in the dectection volume.   
        if ((loca[i,0,n] - 6)**2/Ea**2 + (loca[i,1,n] - 6)**2/Eb**2 + (loca[i,2,n] - 6)**2)/Ec**2 < 1 :
            CB[i,n] = 0
        
        pre_pos[i+1,:,n] = loca[i,:,n]



# In[ ]:
# Calculate the image array when the particles are mobile.
    
image_array = np.zeros((image_size,image_size,len(z_slice)))
image_array_mobile = np.zeros((image_size,image_size,len(z_slice)))


for n in range(Nparticles):

    for k in z_slice: # z
        for j in range(image_array.shape[1]): # x
            for i in range(image_array.shape[0]): # y
                beam_pos = np.array([i,j,k]) * pixel_size
                
                kk = round(k/30 -2)
                particle_pos = loca[ i + image_size * j + image_size*image_size * kk ,:,n]
                image_array[i,j,kk] += CB(i,n) * GaussianBeam(particle_pos,beam_pos,psf_width,psf_height)

image_array_mobile = np.array(image_array)













  
    