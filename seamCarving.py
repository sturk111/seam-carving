import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt

#Image processing functions
#--------------------------

def pixel2ind(x,y,nrows,ncols):
	'''
	Convert the x, y pixel indices of a 2D image with nrows rows and ncols columns into an index into
	the equivalent flattened array 
	'''
	return x*ncols + y

def ind2pixel(i,nrows,ncols):
	'''
	Convert the flattened array index into x, y pixel indices 
	'''
	x = i//ncols
	y = i%ncols
	return x, y


#Graph processing functions
#--------------------------

#topological sort from vertex s in the top row
#downward vertical flow
def topoSort(s,im):
	'''
	A function to generate the list of vertices accessible from a source vertex s in the top row,
	returned in topological order.  See readme for an illustration.

	Parameters
	----------
	s : int
		The index of the source vertex.  Should be between 0 and ncols.
	im : 2D numpy array
		The image for which the topological order is being constructed.  
		This is only used to extract the dimensions.

	Returns
	-------
	topo : list
		The list of vertices accesible from s, returned in topological (equivalent in this case to 
		numerically ascending) order.
	'''
	nrows = im.shape[0]
	ncols = im.shape[1]
	topo = []
	for r in range(nrows):
		left_edge = max(s + r*ncols - r, r*ncols)
		right_edge = min(s + r*ncols + r, r*ncols + ncols - 1)
		for v in range(left_edge,right_edge+1):
			topo.append(v)

	return topo

#Seam carving class
#------------------
class SeamCarver():
	#A class for structuring the seam carving algorithm
	def __init__(self, filepath, scale=1):
		'''
		Import an image from disk and convert to a numpy array.  
		Calculate the "energy" of each pixel as the magnitude of the gradient of adjacent pixels, 
		averaged over color channels.

		Parameters
		----------
		filepath : string
			The filepath where the image to be processed can be found.
		scale : float
			A scale factor used for resizing the image prior to seam carving.  Useful to run faster computations
			on smaller images for testing purposes.
		'''

		#load image from filapath and convert to numpy array
		im = Image.open(filepath)
		tmp = np.array(im).shape
		im = im.resize((int(scale*tmp[1]),int(scale*tmp[0])))
		im = np.array(im)
		self.im = im

		#vectorized calculation of dual-gradient energy function 
		self.energy = 1e3*np.ones((im.shape[0],im.shape[1]))

		Dx2 = np.sum((im[2:,1:-1,:] - im[:-2,1:-1,:])**2,axis=2)
		Dy2 = np.sum((im[1:-1,2:,:] - im[1:-1,:-2,:])**2,axis=2)

		self.energy[1:-1,1:-1] = (Dx2 + Dy2)**0.5
		self.e = self.energy.flatten()


	def nrows(self):
		#returns the number of rows in the image
		return self.im.shape[0]

	def ncols(self):
		#returns the number of columns in the image
		return self.im.shape[1]

	def findVerticalSeam(self):
		'''
		Computes the lowest energy vertical seam using the following procedure.  For each source index in the
		top row (from 0 to ncols), generate a topological order of accessible vertices and relax edges
		pointing from vertices along that order.  Store and return the lowest energy path. 
		This is equivalent to a shortest paths algorithm for a directed acyclic graph with the minor
		modification that weights are associated with vertices rather than edges.

		Returns
		-------
		shortest_path : list
			List of indices of the lowest energy seam.
		seam_img : 2D numpy array
			Array of shape (nrows, ncols) containing ones along the lowest energy seam and
			zeros everywhere else.
		min_store : float
			Length of the lowest energy seam.
		'''
		nrows = self.im.shape[0]
		ncols = self.im.shape[1]
		V = nrows*ncols #total number of vertices

		min_store = np.inf #initialize the shortest path energy to infinity
		for s in range(ncols):
			#initialize arrays for shortest paths algorithm
			distTo = np.full(V,np.inf) 
			edgeTo = np.full(V,np.nan) 
			distTo[s] = self.e[s]

			topOrder = topoSort(s,self.im) #generate topological order from vertex s

			for v in topOrder: 
				#For each vertex v in topological order, define adj, a list of vertices to which v is attached.
				#Each vertex is attached to its nearest neighbors in the row below.

				#handle bottom row
				if v >= ncols*(nrows-1):
					adj = []
				#handle left edge
				elif v%ncols == 0:
					adj = [v+ncols,v+ncols+1]

				#handle right edge
				elif v%ncols == ncols - 1:
					adj = [v+ncols,v+ncols-1]

				#handle all other pixels 
				else:
					adj =[v+ncols-1, v+ncols, v+ncols+1]


				#relax edges in topological order
				for w in adj:
					#print(s,w,v)
					if distTo[w] > distTo[v] + self.e[w]:
						distTo[w] = distTo[v] + self.e[w]
						edgeTo[w] = v

			#record shortest path length to bottom encountered thus far
			min_length = np.min(distTo[ncols*(nrows-1):])
			if min_length < min_store: #if min length from current vertex is shortest so far, update the shortest path
				min_store = min_length
				shortest_path = []
				ptr = int(ncols*(nrows-1) + np.argmin(distTo[ncols*(nrows-1):])) #start from the bottom row
				while not np.isnan(edgeTo[ptr]): #follow the shortest path up to the top row
					shortest_path.append(ptr)
					ptr = int(edgeTo[ptr])
				shortest_path.append(int(s))

			#store the shortest path as an image for later processing
			seam_img = np.zeros((nrows,ncols))
			for i in shortest_path:
				x,y = ind2pixel(i,nrows,ncols)
				seam_img[x,y] = 1

		#return the shortest path
		return shortest_path, seam_img, min_store

	def findHorizontalSeam(self):
		'''
		Computes the lowest energy horizontal seam. Identical to findVerticalSeam, but we first rotate 
		the image by 90 degrees.  After the seam is found, we rotate it back to the original orientation.

		Returns
		-------
		shortest_path : list
			List of indices of the lowest energy seam.
		seam_img : 2D numpy array
			Array of shape (nrows, ncols) containing ones along the lowest energy seam and
			zeros everywhere else.
		min_store : float
			Length of the lowest energy seam.
		'''
		im = np.rot90(self.im) #rotate the raw image
		e = np.rot90(self.energy).flatten() #rotate the energy image

		nrows = im.shape[0]
		ncols = im.shape[1]
		V = nrows*ncols #total number of vertices

		min_store = np.inf
		for s in range(ncols):
			distTo = np.full(V,np.inf)
			edgeTo = np.full(V,np.nan)
			distTo[s] = e[s]

			topOrder = topoSort(s,im)

			for v in topOrder:

				#handle bottom row
				if v >= ncols*(nrows-1):
					adj = []
				#handle left edge
				elif v%ncols == 0:
					adj = [v+ncols,v+ncols+1]

				#handle right edge
				elif v%ncols == ncols - 1:
					adj = [v+ncols,v+ncols-1]

				#handle all other pixels 
				else:
					adj =[v+ncols-1, v+ncols, v+ncols+1]


				#relax edges in topological order
				for w in adj:
					#print(s,w,v)
					if distTo[w] > distTo[v] + e[w]:
						distTo[w] = distTo[v] + e[w]
						edgeTo[w] = v

			#record shortest path length to bottom encountered thus far
			min_length = np.min(distTo[ncols*(nrows-1):])
			if min_length < min_store:
				min_store = min_length
				shortest_path = []
				ptr = int(ncols*(nrows-1) + np.argmin(distTo[ncols*(nrows-1):]))
				while not np.isnan(edgeTo[ptr]):
					shortest_path.append(ptr)
					ptr = int(edgeTo[ptr])
				shortest_path.append(int(s))

		seam_img = np.zeros((nrows,ncols))
		for i in shortest_path:
			x,y = ind2pixel(i,nrows,ncols)
			seam_img[x,y] = 1

		seam_img = np.rot90(seam_img,k=-1) #rotate the seam back to the original orientation
		shortest_path = np.flip(np.nonzero(seam_img.flatten())[0]).tolist() #retrieve indices of the shortest path from the seam image

		#return the shortest path
		return shortest_path, seam_img, min_store

	def removeVerticalSeam(self, seam_img, update = True):
		'''
		Removes a vertical seam from the image and returns an image of size nrows x ncols-1

		Parameters
		----------
		seam_img : 2D numpy array
			Array of shape (nrows, ncols) containing ones along the lowest energy seam and
			zeros everywhere else.  This is the seam to be removed.
		update : bool
			If True then self.im, self.energy, and self.e will be updated to the new reduced size.
			This allows the seam removal to be run repeatedly on the same object, 
			generating an image of progressively reduced size.  

		Returns
		-------
		new_image : numpy array
			Array of shape (nrows, ncols-1, 3) containing the new image with the seam removed.
		'''
		nrows = self.im.shape[0]
		ncols = self.im.shape[1]

		new_image = np.zeros((nrows,ncols-1,3),dtype='uint8') 
		for r in range(nrows):
			new_image[r,:,:] = self.im[r,seam_img[r,:]==0,:] #uses seam_img as a mask to select only those pixels where the seam is not

		if update:
			im = new_image
			self.im = im
			#vectorized calculation of dual-gradient energy function 
			self.energy = 1e3*np.ones((im.shape[0],im.shape[1]))

			Dx2 = np.sum((im[2:,1:-1,:] - im[:-2,1:-1,:])**2,axis=2)
			Dy2 = np.sum((im[1:-1,2:,:] - im[1:-1,:-2,:])**2,axis=2)

			self.energy[1:-1,1:-1] = (Dx2 + Dy2)**0.5
			self.e = self.energy.flatten()


		return new_image

	def removeHorizontalSeam(self, seam_img, update = True):
		'''
		Removes a horizontal seam from the image and returns an image of size nrows-1 x ncols

		Parameters
		----------
		seam_img : 2D numpy array
			Array of shape (nrows, ncols) containing ones along the lowest energy seam and
			zeros everywhere else.  This is the seam to be removed.
		update : bool
			If True then self.im, self.energy, and self.e will be updated to the new reduced size.
			This allows the seam removal to be run repeatedly on the same object, 
			generating an image of progressively reduced size.  

		Returns
		-------
		new_image : numpy array
			Array of shape (nrows-1, ncols, 3) containing the new image with the seam removed.
		'''
		nrows = self.im.shape[0]
		ncols = self.im.shape[1]

		new_image = np.zeros((nrows-1,ncols,3),dtype='uint8')
		for c in range(ncols):
			new_image[:,c,:] = self.im[seam_img[:,c]==0,c,:] #uses seam_img as a mask to select only those pixels where the seam is not

		if update:
			im = new_image
			self.im = im
			#vectorized calculation of dual-gradient energy function 
			self.energy = 1e3*np.ones((im.shape[0],im.shape[1]))

			Dx2 = np.sum((im[2:,1:-1,:] - im[:-2,1:-1,:])**2,axis=2)
			Dy2 = np.sum((im[1:-1,2:,:] - im[1:-1,:-2,:])**2,axis=2)

			self.energy[1:-1,1:-1] = (Dx2 + Dy2)**0.5
			self.e = self.energy.flatten()

		return new_image









			





