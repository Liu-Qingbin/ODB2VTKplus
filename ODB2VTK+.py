#-*-coding:UTF-8-*-
'''
The script is developed by Qingbin Liu
E-mail: liuqingb@pku.edu.cn
'''

#import necessary modules which support operations
from odbAccess import *
from textRepr import *
#from string import *
from time import *
from math import *
import copy
import multiprocessing
import numpy as np

#Dictionary of label and name of variables
#It can be modified if necessary
VARIABLEDICT = {'U':"Spatial_displacement", 'A':"Spatial_acceleration", 'V':"Spatial_velocity",'RF':"Reaction_force",\
'S':"Stress", 'LE':"Logarithmic_strain", 'PE':"Plastic_strain",'PEEQ':"Equivalent_plastic_strain",\
'MISES':"Mises", 'MAX_PRINCIPAL':"Max_Principal", 'MID_PRINCIPAL':"Mid_Principal", 'MIN_PRINCIPAL':"Min_Principal",\
'TRESCA':"Tresca", 'PRESS':"Pressure", 'INV3':"Third_Invariant"}


###==============================Sub=Function=============================###
#-----------------------------Read Configuration----------------------------#

#To get the parameters from configuration file 
def GetConfig(filename = 'odb2vtk.cfg'):	
	# Check if filename points to existing file
	if not os.path.isfile(filename): 
		print 'Parameter file "%s" not found'%(filename)
		sys.exit(2) 
	
	#get parameters from configuration file
	config = open(filename,'r')
	read = config.read()
	input = read.split("'")
	#get odb file's path
	odb_path = input[1]
	#get odb file's name
	odb_name = input[3]
	#get the output vtk files' path
	vtk_path = input[5]
	#get the mesh type ( ex: Hexahedron, Tetrahedron, etc. )
	mesh_type = input[7]
	mesh_conner = 0
	if (mesh_type == 'Hexahedron'):
		mesh_conner = 8
		mesh_type = 12
		mesh_name = "Hexahedron"
	if (mesh_type == 'Tetrahedron'):
		mesh_conner = 4
		mesh_type = 10
		mesh_name = "Tetrahedron"
	if (mesh_conner == 0):
		print("Mesh type error or unidentified")
		os._exit(0)
	#get the selection of partition method (ex: Random, Regular, CenterPoint, etc.)
	partition_method = input[9]
	#get the parameters of partition, such as quantity and its distribution
	temp_partition_parameter = input[11]
	if(partition_method == 'Random'):
		partition_parameter = int(temp_partition_parameter)
		piece_number = int(temp_partition_parameter)
	elif(partition_method == 'Regular'):
		partition_parameter = temp_partition_parameter.split(',')
		piece_number = int(partition_parameter[0]) * int(partition_parameter[1]) * int(partition_parameter[2])
	elif(partition_method == 'CenterPoint'):
		temppoint = temp_partition_parameter.split(',')
		piece_number = int(len(temppoint) / 3)
		partition_parameter = np.zeros(piece_number * 3).reshape(piece_number ,3)
		for i in range(piece_number):
			for j in range(3):
				partition_parameter[i][j] = float(temppoint[i * 3 + j])
	else:
		print "partition method " + partition_method + " doesn't exist, it must be 'Random','Regular' or 'CenterPoint', Please Check!"
		os._exit(0)
	#get the frame to transfer
	input_frame = input[13].split("-")
	input_frame = range(int(input_frame[0]),int(input_frame[1])+1)
	#get the step to transfer
	input_step = input[15].split(",")
	#get the instance to transfer
	input_instance = input[17].split(",")
	#get the variables to transfer
	output_scalar = input[19].split(",")
	output_vector = input[21].split(",")
	output_tensor = input[23].split(",")
	#get the multiprocessing set
	parallel_partition = input[25]
	partition_processing_number = int(input[27])
	parallel_frame = input[29]
	frame_processing_number = int(input[31])
	
	#end reding and close odb2vtk file
	config.close()

	#display the reading result of odb2vtk file
	print "Basic Information:" 
	print "Model:",odb_name,"; Mesh type:",mesh_name
	print "Partition method:",partition_method,"; partition_parameter:",partition_parameter
	print "Convert frames: ",input_frame[0]," to ",input_frame[-1]
	print "Step & Instance : ",str(input_step),", ",str(input_instance)
	print "Parallel partition : ", parallel_partition
	print "parallel_frame : ", parallel_frame
	
	return{'odb_path':odb_path,'odbname':odb_name,'vtk_path':vtk_path,'mesh_type':mesh_type,'mesh_conner':mesh_conner,\
	'partition_method':partition_method,'partition_parameter':partition_parameter,'piece_number':piece_number,\
	'input_frame':input_frame,'input_step':input_step,'input_instance':input_instance,
	'output_scalar':output_scalar,'output_vector':output_vector,'output_tensor':output_tensor,
	'parallel_partition':parallel_partition,'partition_processing_number':partition_processing_number,\
	'parallel_frame':parallel_frame,'frame_processing_number':frame_processing_number} 


#-----------------------------Read Configuration----------------------------#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#------------------------------Read Attribution-----------------------------#

#To Read nodes' coordination, return a (n*3)numpy
def ReadNode(node = '', max_nodelabel = 0):
	node_coordination = np.zeros(max_nodelabel*3).reshape(max_nodelabel,3)
	for i_node in node:
		node_coordination[i_node.label-1] = i_node.coordinates
	return node_coordination


#To read scalar variable, return a (1*n)numpy	
def ReadScalarAtInt(odbscalar = '',outputlength = 0):
	nodal_fieldvalues = odbscalar.getSubset(position=ELEMENT_NODAL)
	fieldValues = nodal_fieldvalues.values
	acount = np.zeros(outputlength)
	scalaroutput = np.zeros(outputlength)
	for scalar_value in fieldValues :
		acount[scalar_value.nodeLabel-1] += 1
		scalaroutput[scalar_value.nodeLabel-1] += scalar_value.data
	for i in range(outputlength):
		if(acount[i] != 0):
			scalaroutput[i] = scalaroutput[i]/acount[i]
	return scalaroutput


#To read vector variable, return a (n*3)numpy	
def ReadVector(odbvector = '',outputlength = 0):
	fieldvalues = odbvector.values
	vectoroutput = np.zeros(outputlength*3).reshape(outputlength,3) 
	for vector_value in fieldvalues :
		vectoroutput[vector_value.nodeLabel-1] = vector_value.data
	return vectoroutput

	
#To read tensor vriable, transfer the variable from integration point to nodal point,
#return a dicrtionary with two numpy:
#the first is tensor(), the second is invariant()
#modify the function can read tensor variable with different invariant
def ReadTensorAtIntS(odbtensor = '',invariantamount = 0,outputlength = 0):
	nodal_fieldvalues = odbtensor.getSubset(position=ELEMENT_NODAL)
	fieldvalues = nodal_fieldvalues.values
	acount = np.zeros(outputlength)
	outputtensor = np.zeros(outputlength*6).reshape(outputlength,6)
	outputinvariant = np.zeros(outputlength*invariantamount).reshape(outputlength,invariantamount)
	for tensor_value in fieldvalues:
		acount[tensor_value.nodeLabel-1] += 1
		outputtensor[tensor_value.nodeLabel-1] += tensor_value.data
		#modification!!!
		outputinvariant[tensor_value.nodeLabel-1][0] += tensor_value.mises
		outputinvariant[tensor_value.nodeLabel-1][1] += tensor_value.maxPrincipal
		outputinvariant[tensor_value.nodeLabel-1][2] += tensor_value.midPrincipal
		outputinvariant[tensor_value.nodeLabel-1][3] += tensor_value.minPrincipal
		outputinvariant[tensor_value.nodeLabel-1][4] += tensor_value.press
		outputinvariant[tensor_value.nodeLabel-1][5] += tensor_value.tresca
		outputinvariant[tensor_value.nodeLabel-1][6] += tensor_value.inv3
	for i in range(outputlength):
		if(acount[i] != 0):
			outputtensor[i] = outputtensor[i]/acount[i]
			outputinvariant[i] = outputinvariant[i]/acount[i]
	listinvariant = ['MISES', 'MAX_PRINCIPAL', 'MID_PRINCIPAL', 'MIN_PRINCIPAL', 'TRESCA', 'PRESS', 'INV3']
	return {'tensor':outputtensor,'invariant':outputinvariant,'validinvariant':listinvariant}

def ReadTensorAtIntE(odbtensor = '',invariantamount = 0,outputlength = 0):
	nodal_fieldvalues = odbtensor.getSubset(position=ELEMENT_NODAL)
	fieldvalues = nodal_fieldvalues.values
	acount = np.zeros(outputlength)
	outputtensor = np.zeros(outputlength*6).reshape(outputlength,6)
	outputinvariant = np.zeros(outputlength*invariantamount).reshape(outputlength,invariantamount)
	for tensor_value in fieldvalues:
		acount[tensor_value.nodeLabel-1] += 1
		outputtensor[tensor_value.nodeLabel-1] += tensor_value.data
		#modification!!!
		outputinvariant[tensor_value.nodeLabel-1][0] += tensor_value.maxPrincipal
		outputinvariant[tensor_value.nodeLabel-1][1] += tensor_value.minPrincipal
	for i in range(outputlength):
		if(acount[i] != 0):
			outputtensor[i] = outputtensor[i]/acount[i]
			outputinvariant[i] = outputinvariant[i]/acount[i]
	listinvariant = ['MAX_PRINCIPAL', 'MIN_PRINCIPAL']
	return {'tensor':outputtensor,'invariant':outputinvariant,'validinvariant':listinvariant}


#------------------------------Read Attribution-----------------------------#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#-------------------------Partition & Reorganization------------------------#

#---------------------Random Partition & Reorganization---------------------#
#Partition model into pieces by the label of element which might be random
def PartitionRandom(config = {}, element_amount = 0):
	piecenum = config['piece_number']
	element_piece_amount = element_amount/piecenum
	last_element_piece_amount = element_amount - (element_piece_amount*(piecenum-1))  #last block
	piecelist = []
	for i in range(piecenum):
		piecelist.append(str(i))
	piecedict = dict.fromkeys(piecelist)	
	for i_piece in range(piecenum):
		if(i_piece == piecenum-1):
			piecedict[str(i_piece)] = copy.deepcopy(np.arange(i_piece*element_piece_amount,element_amount))
		else:
			piecedict[str(i_piece)] = copy.deepcopy(np.arange(i_piece*element_piece_amount,(i_piece+1)*element_piece_amount))
	return piecedict


#Reorganize model partitioned by random method in serial
def ReorganizeRandomNodeElementSerial(config = {}, piecedict = {},element = '', max_nodelabel = 0):
	mesh_conner = config['mesh_conner']
	piecelist = piecedict.keys()
	newmodel = dict.fromkeys(piecelist)
	#reorganize the node and element (reconstruct the mesh)
	for i_keys in piecelist:
		#nodeexisted: mark the nodes which have already existed in new model
		nodeexisted = np.zeros(max_nodelabel,dtype=np.int32)
		nodeexisted = -1
		#elementnode: store the reorganized element
		elementnode = np.zeros(0,dtype=np.int32)
		#newnode: store the reorganized node
		newnode = np.zeros(0,dtype=np.int32)
		nodecount = 0
		#reorganize each piece
		i_piece = piecedict[i_keys]
		for i in i_piece:
			i_element = element[i]
			for j in range(mesh_conner):
				k = i_element.connectivity[j] - 1
				if(nodeexisted[k] < 0):
					nodeexisted[k] = nodecount
					newnode = np.append(newnode,k)
					elementnode = np.append(elementnode,nodecount)
					nodecount += 1
				else:
					elementnode = np.append(elementnode,nodeexisted[k])
		newmodel[i_keys] = {'node':copy.deepcopy(newnode),'element':copy.deepcopy(elementnode)}
	return newmodel	


#Reorganize model partitioned by random method in parallel
def ReorganizeRandomNodeElementParallel(config = {}, piecedict = {}, instancename = '', max_nodelabel = 0):
	piecelist = piecedict.keys()
	manager = multiprocessing.Manager()
	newmodel =  manager.dict()
	#reorganize the node and element (reconstruct the mesh)
	#control the process
	piecelist_length = len(piecelist)
	process_munber = config['partition_processing_number']
	piece_each_process = piecelist_length/process_munber
	control_dict = {}
	#distribute pieces to process
	for i in range(process_munber-1):
		control_dict[str(i)] = piecelist[i*piece_each_process:(i+1)*piece_each_process]
	control_dict[str(process_munber-1)]	= piecelist[(process_munber-1)*piece_each_process : piecelist_length]
	print "Partition by multiprocessing......"
	processlist = []
	for i in range(process_munber):
		P = multiprocessing.Process(target = ReorganizeRandomNodeElementSubFunction,\
		args = (newmodel,piecedict,control_dict[str(i)],config,instancename,max_nodelabel))
		processlist.append(P)
		P.start()
		print "process-ID: ",P.pid
		
	for i in processlist:
		P.join()
	
	for i in processlist:
		P.terminate()	
		
	return newmodel		

def ReorganizeRandomNodeElementSubFunction(newmodel,piecedict,controllist,config,instancename,max_nodelabel):
	odb = openOdb(os.path.join(config['odb_path'],config['odbname'])+'.odb',readOnly=True)
	element = odb.rootAssembly.instances[instancename].elements
	for i_keys in controllist:
		#nodeexisted: mark the nodes which have already existed in new model
		nodeexisted = np.zeros(max_nodelabel,dtype=np.int32)
		nodeexisted[:] = -1
		#elementnode: store the reorganized element
		elementnode = np.zeros(0,dtype=np.int32)
		#newnode: store the reorganized node
		newnode = np.zeros(0,dtype=np.int32)
		nodecount = 0
		#reorganize each piece
		i_piece = piecedict[i_keys]
		for i in i_piece:
			i_element = element[i]
			for j in range(config['mesh_conner']):
				k = i_element.connectivity[j] - 1
				if(nodeexisted[k] < 0):
					nodeexisted[k] = nodecount
					newnode = np.append(newnode,k)
					elementnode = np.append(elementnode,nodecount)
					nodecount += 1
				else:
					elementnode = np.append(elementnode,nodeexisted[k])
		newmodel[i_keys] = {'node':copy.deepcopy(newnode),'element':copy.deepcopy(elementnode)}


#-------------------------------Regular Partition---------------------------#
#Partition model into pieces by regular partitoin method in serial
def PartitionRegularSerial(config = {},node_coordination = [], element = ''):
	
	piecesize = config['partition_parameter']
	mesh_conner = config['mesh_conner']

	x_size = int(piecesize[0])
	y_size = int(piecesize[1])
	z_size = int(piecesize[2])
	
	node_coordination = node_coordination.T
	x_max = np.max(node_coordination[0])
	x_min = np.min(node_coordination[0])
	y_max = np.max(node_coordination[1])
	y_min = np.min(node_coordination[1])
	z_max = np.max(node_coordination[2])
	z_min = np.min(node_coordination[2])
	node_coordination = node_coordination.T
	
	x_interval = (x_max - x_min) / x_size
	y_interval = (y_max - y_min) / y_size
	z_interval = (z_max - z_min) / z_size
	
	x_limit = np.zeros(x_size + 1)
	y_limit = np.zeros(y_size + 1)
	z_limit = np.zeros(z_size + 1)
	for i in range(x_limit.shape[0]):
		x_limit[i] = x_min + (i * x_interval)
	x_limit[0] = x_min - 10
	x_limit[x_limit.shape[0] - 1] = x_max + 10
	for i in range(y_limit.shape[0]):
		y_limit[i] = y_min + (i * y_interval)
	y_limit[0] = y_min - 10
	y_limit[y_limit.shape[0] - 1] = y_max + 10
	for i in range(z_limit.shape[0]):
		z_limit[i] = z_min + (i * z_interval)
	z_limit[0] = z_min - 10
	z_limit[z_limit.shape[0] - 1] = z_max + 10
	
	#create the dictionary of partition piece, storing the element label in each piece
	piecelist = []
	for i in range(x_size):
		for j in range(y_size):
			for k in range(z_size):
				piecelist.append(str(i)+str(j)+str(k))
	piecedict = dict.fromkeys(piecelist)
	for i in range(x_size):
		for j in range(y_size):
			for k in range(z_size):
				piecedict[str(i)+str(j)+str(k)] = copy.deepcopy(np.zeros(0))
	#create the numpy for storing the connectivity of each element
	element_connectivity = np.zeros(len(element)*mesh_conner,dtype=np.int32).reshape(len(element),mesh_conner)
	
	tempcoordinate = np.zeros(3)
	for i_element in element:
		label = i_element.label
		element_connectivity[label - 1] = i_element.connectivity
		tempcoordinate[:] = 0
		for elementnode in range(mesh_conner):
			k = i_element.connectivity[elementnode] - 1
			tempcoordinate += node_coordination[k]
		tempcoordinate = tempcoordinate / mesh_conner
		
		i = 0
		j = x_size
		x = tempcoordinate[0]
		while( (j - i) > 1):
			k = (i + j) / 2
			if (x > x_limit[k]):
				i = k
			else:
				j = k
		i_x = i
		
		i = 0
		j = y_size
		y = tempcoordinate[1]
		while( (j - i) > 1):
			k = (i + j) / 2
			if (y > y_limit[k]):
				i = k
			else:
				j = k
		i_y = i
		
		i = 0
		j = z_size
		z = tempcoordinate[2]
		while( (j - i) > 1):
			k = (i + j) / 2
			if (z > z_limit[k]):
				i = k
			else:
				j = k
		i_z = i
		
		tempkey = str(i_x)+str(i_y)+str(i_z)
		piecedict[tempkey] = np.append(piecedict[tempkey], label)
	
	return piecedict,element_connectivity


#Partition model into pieces by regular partitoin method in parallel
def PartitionRegularParallel(config = {}, node_coordination = [], instancename = ''):
	odb = openOdb(os.path.join(config['odb_path'],config['odbname'])+'.odb',readOnly=True)
	element = odb.rootAssembly.instances[instancename].elements
	mesh_conner = config['mesh_conner']
	
	piecesize = config['partition_parameter']
	x_size = int(piecesize[0])
	y_size = int(piecesize[1])
	z_size = int(piecesize[2])
	
	node_coordination = node_coordination.T
	x_max = np.max(node_coordination[0])
	x_min = np.min(node_coordination[0])
	y_max = np.max(node_coordination[1])
	y_min = np.min(node_coordination[1])
	z_max = np.max(node_coordination[2])
	z_min = np.min(node_coordination[2])
	node_coordination = node_coordination.T
	
	x_interval = (x_max - x_min) / x_size
	y_interval = (y_max - y_min) / y_size
	z_interval = (z_max - z_min) / z_size
	
	x_limit = np.zeros(x_size + 1)
	y_limit = np.zeros(y_size + 1)
	z_limit = np.zeros(z_size + 1)
	for i in range(x_limit.shape[0]):
		x_limit[i] = x_min + (i * x_interval)
	x_limit[0] = x_min - 10
	x_limit[x_limit.shape[0] - 1] = x_max + 10
	for i in range(y_limit.shape[0]):
		y_limit[i] = y_min + (i * y_interval)
	y_limit[0] = y_min - 10
	y_limit[y_limit.shape[0] - 1] = y_max + 10
	for i in range(z_limit.shape[0]):
		z_limit[i] = z_min + (i * z_interval)
	z_limit[0] = z_min - 10
	z_limit[z_limit.shape[0] - 1] = z_max + 10
	
	#create the dictionary of partition piece, storing the element label in each piece
	manager = multiprocessing.Manager()
	piecedict =  manager.dict()
	for i in range(x_size):
		for j in range(y_size):
			for k in range(z_size):
				piecedict[str(i)+str(j)+str(k)] = copy.deepcopy(np.zeros(0,dtype=np.int32))
	element_length = len(element)
	elementdict = manager.dict()
	elementpiecedict = manager.dict()

	#control the process
	process_munber = config['partition_processing_number']
	element_each_process = element_length/process_munber
	control_dict = {}
	#distribute elements to process
	for i in range(process_munber-1):
		control_dict[str(i)] = range(i*element_each_process, (i+1)*element_each_process)
	control_dict[str(process_munber-1)]	= range((process_munber-1)*element_each_process, element_length)
	
	processlist = []
	for i in range(process_munber):
		P = multiprocessing.Process(target = PartitionRegularSubfunction, args = (config,elementpiecedict,elementdict,control_dict[str(i)],node_coordination,instancename,x_limit,y_limit,z_limit,x_size,y_size,z_size))
		processlist.append(P)
		P.start()
		print "process-ID: ",P.pid
	
	for i in processlist:
		P.join()
	
	for i in processlist:
		P.terminate()
	
	print('Parallel Finish')

	element_connectivity = np.zeros(element_length*mesh_conner,dtype=np.int32).reshape(element_length,mesh_conner)
	for i in range(element_length):
		element_connectivity[i] = elementdict[str(i)]
		tmpkey = elementpiecedict[str(i)]
		piecedict[tmpkey] = np.append(piecedict[tmpkey] , i + 1)
	
	print('return piecedict,element_connectivity')
	return piecedict, element_connectivity

def PartitionRegularSubfunction(config,elementpiecedict,elementdict,controllist,node_coordination,instancename,x_limit,y_limit,z_limit,x_size,y_size,z_size):
	odb = openOdb(os.path.join(config['odb_path'],config['odbname'])+'.odb',readOnly=True)
	element = odb.rootAssembly.instances[instancename].elements
	mesh_conner = config['mesh_conner']

	tempcoordinate = np.zeros(3)
	for ie_label in controllist:
		i_element = element[ie_label]
		label = i_element.label
		elementdict[str(label - 1)] = i_element.connectivity
		tempcoordinate[:] = 0
		for elementnode in range(mesh_conner):
			k = i_element.connectivity[elementnode] - 1
			tempcoordinate += node_coordination[k]
		# get the center point coordinates of element
		tempcoordinate = tempcoordinate / mesh_conner
		
		i = 0
		j = x_size
		x = tempcoordinate[0]
		while( (j - i) > 1):
			k = (i + j) / 2
			if (x > x_limit[k]):
				i = k
			else:
				j = k
		i_x = i
		
		i = 0
		j = y_size
		y = tempcoordinate[1]
		while( (j - i) > 1):
			k = (i + j) / 2
			if (y > y_limit[k]):
				i = k
			else:
				j = k
		i_y = i
		
		i = 0
		j = z_size
		z = tempcoordinate[2]
		while( (j - i) > 1):
			k = (i + j) / 2
			if (z > z_limit[k]):
				i = k
			else:
				j = k
		i_z = i
		
		tempkey = str(i_x)+str(i_y)+str(i_z)
		elementpiecedict[str(label - 1)] = tempkey


#-----------------------------CenterPoint Partition-------------------------#
#Partition model into pieces by indentifing the center point of the piece in serial
def PartitionPointCenterSerial(config = {}, node_coordination = [], element = ''):
	
	centerpoint = config['partition_parameter']
	mesh_conner = config['mesh_conner']
	piece_number = config['piece_number']
	
	for i in range(piece_number):
		piecelist = piecelist.append(str(i))
	for i in piecelist:
		piecedict[i] = copy.deepcopy(np.zeros(0))
	
	element_connectivity = np.zeros(len(element)*mesh_conner,dtype=np.int32).reshape(len(element),mesh_conner)
	
	tempcoordinate = np.zeros(3)
	tempdistance = np.zeros(piecenum)
	tempkey = ''
	for i_element in element:
		label = i_element.label
		element_connectivity[label - 1] = i_element.connectivity
		tempcoordinate[:] = 0
		for elementnode in range(mesh_conner):
			k = i_element.connectivity[elementnode] - 1
			tempcoordinate += node_coordination[k]
		tempcoordinate = tempcoordinate / mesh_conner
		for i in range(piecenum):
			tempdistance[i] = PointDistance(tempcoordinate,centerpoint[i])
		tempkey = np.argmin(tempdistance)
		tempkey = str(tempkey)
		piecedict[tempkey] = np.append(piecedict[tempkey], label)
		
	return piecedict,element_connectivity


#Partition model into pieces by indentifing the center point of the piece in parallel
def PartitionPointCenterParallel(config = {}, node_coordination = [], instancename = ''):
	#Open ODB
	odb = openOdb(os.path.join(config['odb_path'],config['odbname'])+'.odb',readOnly=True)
	element = odb.rootAssembly.instances[instancename].elements
	
	element_length = len(element)
	centerpoint = config['partition_parameter']
	mesh_conner = config['mesh_conner']
	piece_number = config['piece_number']
	
	manager = multiprocessing.Manager()
	piecedict =  manager.dict()
	
	for i in range(piece_number):
		piecelist = piecelist.append(str(i))
	for i in piecelist:
		piecedict[i] = copy.deepcopy(np.zeros(0))
	
	elementdict = manager.dict()
	for i in range(len(element)):
		elementdict[str(i)] = copy.deepcopy(np.zeros(mesh_conner,dtype=np.int32))
	
	#control the process
	process_munber = config['partition_processing_number']
	element_each_process = element_length/process_munber
	control_dict = {}
	#distribute elements to process
	for i in range(process_munber-1):
		control_dict[str(i)] = range(i*element_each_process, (i+1)*element_each_process)
	control_dict[str(process_munber-1)]	= range((process_munber-1)*element_each_process, element_length)
	processlist = []
	for i in range(process_munber):
		P = multiprocessing.Process(target = PartitionPointCenterSubfunction, \
		args = (config,piecedict,elementdict,control_dict[str(i)],node_coordination,instancename,centerpoint))
		processlist.append(P)
		P.start()
		print "process-ID: ",P.pid
	
	for i in processlist:
		P.join()
	
	for i in processlist:
		P.terminate()
	
	
	element_connectivity = np.zeros(element_length*mesh_conner,dtype=np.int32).reshape(element_length,mesh_conner)
	for i in range(element_length):
		element_connectivity[i] = elementdict[str(i)]
	
	print('return piecedict,element_connectivity')	
	return piecedict,element_connectivity

def PartitionPointCenterSubfunction(config,piecedict,elementdict,controllist,node_coordination,instancename,centerpoint):
	odb = openOdb(os.path.join(config['odb_path'],config['odbname'])+'.odb',readOnly=True)
	element = odb.rootAssembly.instances[instancename].elements
	mesh_conner = config['mesh_conner']
	piece_number = config['piece_number']
	
	tempcoordinate = np.zeros(3)
	tempdistance = np.zeros(piece_number)
	tempkey = ''
	for ie_label in controllist:
		i_element = element[ie_label]
		label = i_element.label
		elementdict[label - 1] = i_element.connectivity
		tempcoordinate[:] = 0
		for elementnode in range(mesh_conner):
			k = i_element.connectivity[elementnode] - 1
			tempcoordinate += node_coordination[k]
		tempcoordinate = tempcoordinate / mesh_conner
		for i in range(piece_number):
			tempdistance[i] = PointDistance(tempcoordinate,centerpoint[i])
		tempkey = np.argmin(tempdistance)
		tempkey = str(tempkey)
		piecedict[tempkey] = np.append(piecedict[tempkey], label)

def PointDistance(Point1 = [], Point2 = []):
	return sqrt(np.sum(np.square(Point1-Point2)))		


#------------------------------Reorganization------------------------------#
#Reorganize partitioned model in serial
def ReorganizeNodeElementSerial(config = {}, piecedict = {}, element_connectivity = [], max_nodelabel = 0):
	mesh_conner = config['mesh_conner']
	piecelist = piecedict.keys()
	newmodel = dict.fromkeys(piecelist)	
	#reorganize the node and element (reconstruct the mesh)
	for i_keys in piecelist:
		i_piece = piecedict[i_keys]
		#nodeexisted: estimate whether the node has already existed
		nodeexisted = np.zeros(max_nodelabel, dtype=np.int32)
		nodeexisted[:] = -1
		#elementnode: store the reorganized node for element
		elementnode = np.zeros(0,dtype=np.int32)
		#newnode: store the reorganized node for node
		newnode = np.zeros(0,dtype=np.int32)
		nodecount = 0
		for i in i_piece:
			connectivity = element_connectivity[i-1]
			for j in range(mesh_conner):
				k = connectivity[j] - 1
				if(nodeexisted[k] < 0):
					nodeexisted[k] = nodecount
					newnode = np.append(newnode,k)
					elementnode = np.append(elementnode,nodecount)
					nodecount += 1
				else:
					elementnode = np.append(elementnode,nodeexisted[k])
		newmodel[i_keys] = {'node':copy.deepcopy(newnode),'element':copy.deepcopy(elementnode)}
	return newmodel


#Reorganize partitioned model in parallel
def ReorganizeNodeElementParallel(config = {}, piecedict = {}, element_connectivity = [], max_nodelabel = 0):
	piecelist = piecedict.keys()
	manager = multiprocessing.Manager()
	newmodel =  manager.dict()
	#reorganize the node and element (reconstruct the mesh)
	#control the process
	piecelist_length = len(piecelist)
	process_munber = config['partition_processing_number']
	piece_each_process = piecelist_length/process_munber
	control_dict = {}
	#distribute pieces to process
	for i in range(process_munber-1):
		control_dict[str(i)] = piecelist[i*piece_each_process:(i+1)*piece_each_process]
	control_dict[str(process_munber-1)]	= piecelist[(process_munber-1)*piece_each_process : piecelist_length]
	print "Reorganize by multiprocessing......"
	processlist = []
	for i in range(process_munber):
		P = multiprocessing.Process(target = ReorganizeNodeElementSubfunction,\
		args = (newmodel,piecedict,element_connectivity,control_dict[str(i)],config,max_nodelabel))
		processlist.append(P)
		P.start()
		print "process-ID: ",P.pid
		
	for i in processlist:
		P.join()
	
	for i in processlist:
		P.terminate()	
	
	return newmodel		

def ReorganizeNodeElementSubfunction(newmodel, piecedict, element_connectivity, controllist, config, max_nodelabel):
	mesh_conner = config['mesh_conner']
	#reorganize the node and element (reconstruct the mesh)
	for i_keys in controllist:
		i_piece = piecedict[i_keys]
		#nodeexisted: estimate whether the node has already existed
		nodeexisted = np.zeros(max_nodelabel, dtype=np.int32)
		nodeexisted[:] = -1
		#elementnode: store the reorganized node for element
		elementnode = np.zeros(0,dtype=np.int32)
		#newnode: store the reorganized node for node
		newnode = np.zeros(0,dtype=np.int32)
		nodecount = 0
		for i in i_piece:
			connectivity = element_connectivity[i-1]
			for j in range(mesh_conner):
				k = connectivity[j] - 1
				if(nodeexisted[k] < 0):
					nodeexisted[k] = nodecount
					newnode = np.append(newnode,k)
					elementnode = np.append(elementnode,nodecount)
					nodecount += 1
				else:
					elementnode = np.append(elementnode,nodeexisted[k])
		newmodel[i_keys] = {'node':copy.deepcopy(newnode),'element':copy.deepcopy(elementnode)}


#-------------------------Partition & Reorganization------------------------#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#---------------------Read Attribution and Output to VTK--------------------#
#Read attribution from each frame and write to VTK file format in serial
def WriteToVTKSerial(config = {}, frame = '', newmodel = {}, node_coordination = [], stepname = '', instancename = '',  max_nodelabel = 0):
	frame_list = config['input_frame']
	for i_frame in frame_list:
		#Detect whether the input frame is out of range
		try:
			TRY = frame[int(i_frame)]
		except:
			print "input frame exceeds the range of frames" 
			os._exit(0)
		
		#Access a frame
		n_frame = frame[int(i_frame)]
		print "Frame:",i_frame


		#Access Vector: Spatial displacement, acceleration, velocity and reaction force
		vectorlist = config['output_vector']
		print "Reading"+str(vectorlist)+" ......"
		time1 = time()
		vector_amount = len(vectorlist)
		vector_dict = dict.fromkeys(vectorlist)
		for vector_name in vectorlist:
			tempvector = ReadVector(n_frame.fieldOutputs[vector_name],max_nodelabel)
			vector_dict[vector_name] = copy.deepcopy(tempvector)
		print "Time elapsed: ", time() - time1, "s"
		
		#modification
		#Access Tensor: Stress components, Logarithmic strain components and Plastic strain components
		tensorlist = config['output_tensor']
		tensor_dict = dict.fromkeys(tensorlist)
		print "Reading"+str(tensorlist)+"......"
		time1 = time()
		#access Stress components
		tensor_dict[tensorlist[0]] = ReadTensorAtIntS(n_frame.fieldOutputs[tensorlist[0]],7,max_nodelabel)
		#Logarithmic strain components
		tensor_dict[tensorlist[1]] = ReadTensorAtIntE(n_frame.fieldOutputs[tensorlist[1]],2,max_nodelabel)
		#Plastic strain components
		tensor_dict[tensorlist[2]] = ReadTensorAtIntE(n_frame.fieldOutputs[tensorlist[2]],2,max_nodelabel)
		print "Time elapsed: ", time() - time1, "s"
		
		#Access Saclar: Equivalent plastic strain
		saclarlist = config['output_scalar']
		scalar_dict = dict.fromkeys(saclarlist)
		print "Reading"+str(saclarlist)+"......"
		#Equivalent plastic strain
		time1 = time()
		scalar_dict[saclarlist[0]] = ReadScalarAtInt(n_frame.fieldOutputs[saclarlist[0]],max_nodelabel)
		print "Time elapsed: ", time() - time1, "s"	
		
		
		
		piecekey = newmodel.keys()
		for pn in piecekey:
			print "Create .vtu file for piece " + pn + "......"
			#create and open a VTK(.vtu) files
			if(config['piece_number'] > 1):
				outfile = open (os.path.join(config['vtk_path'],config['odbname'])+'_'+stepname+'_'+instancename+'f%03d'%int(i_frame)+' '+'p'+str(pn)+'.vtu','w')
			if(config['piece_number'] == 1):
				outfile = open (os.path.join(config['vtk_path'],config['odbname'])+'_'+stepname+'_'+instancename+'f%03d'%int(i_frame)+'.vtu','w')
			
			#<VTKFile>, including the type of mesh, version, and byte_order
			outfile.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">'+'\n')
			#<UnstructuredGrid>
			outfile.write('<UnstructuredGrid>'+'\n')
			
			
			#get the partitioned model
			newnode = newmodel[pn]['node']
			newelement = newmodel[pn]['element']
			#compute point & element quantity
			newelementamount = newelement.shape[0] / config['mesh_conner']
			newnodeamount = newnode.shape[0]
			newnodeamount_N = range(0,newnodeamount)
			#<Piece>, including the number of points and cells
			outfile.write('<Piece NumberOfPoints="'+str(newnodeamount)+'"'+' '+'NumberOfCells="'+str(newelementamount)+'">'+'\n')
			
			
			
			#<Points> Write nodes into vtk files
			tempvector = vector_dict['U']
			outfile.write('<Points>'+'\n')
			outfile.write('<DataArray type="Float64" NumberOfComponents="3" format="ascii">'+'\n')
			for i in newnodeamount_N:
				k = newnode[i]
				X,Y,Z = node_coordination[k]+tempvector[k]
				outfile.write(' '+'%11.8e'%X+'  '+'%11.8e'%Y+'  '+'%11.8e'%Z+'\n')			
			outfile.write('</DataArray>'+'\n')
			outfile.write('</Points>'+'\n')
			#</Points>

			
			#<PointData> Write results data into vtk files
			templist = []
			templist.append("<PointData ")
			templist.append("Tensors="+'"')
			for tensorkey in tensorlist:
				templist.append(VARIABLEDICT[tensorkey])
				templist.append(',')
			templist.pop()
			templist.append('"'+" ")
			templist.append("Vevtors="+'"')
			for vectorkey in vectorlist:
				templist.append(VARIABLEDICT[vectorkey])
				templist.append(',')
			templist.pop()
			templist.append('"'+" ")
			templist.append("Scalars="+'"')
			for scalarkey in saclarlist:
				templist.append(VARIABLEDICT[scalarkey])
				templist.append(',')
			for tensorkey in tensorlist:
				for invariantkey in tensor_dict[tensorkey]['validinvariant']:
					templist.append(VARIABLEDICT[tensorkey]+VARIABLEDICT[invariantkey])
					templist.append(',')
			templist.append('"'+">"+'\n')
			outfile.write("".join(templist))
			
			#Tensor: Stress components, Logarithmic strain components, Plastic strain components <DataArray>
			for tensorkey in tensorlist:
				temptensor = tensor_dict[tensorkey]['tensor']
				outfile.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[tensorkey]+'"'+" "+"NumberOfComponents="+'"'+"9"+'"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
				for i in newnodeamount_N:
					k = newnode[i]
					#XX,XY,XZ,YY,YZ,ZZ = temptensor[k][0],temptensor[k][3],temptensor[k][5],temptensor[k][1],temptensor[k][4],temptensor[k][2]
					XX,YY,ZZ,XY,YZ,XZ = temptensor[k]
					YX,ZY,ZX = temptensor[k][3], temptensor[k][4], temptensor[k][5]
					outfile.write('%11.8e'%XX+' '+'%11.8e'%XY+' '+'%11.8e'%XZ+' '+'%11.8e'%YX+' '+'%11.8e'%YY+' '+'%11.8e'%YZ+' '+'%11.8e'%ZX+' '+'%11.8e'%ZY+' '+'%11.8e'%ZZ+'\n')
				outfile.write("</DataArray>"+'\n')
			#</DataArray>
			
			#Vector: Spatial displacement, acceleration, velocity, Reaction force <DataArray>
			for vectorkey in vectorlist:
				tempvector = vector_dict[vectorkey]
				outfile.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[vectorkey]+'"'+" "+"NumberOfComponents="+'"'+"3"+'"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
				for i in newnodeamount_N:
					k = newnode[i]
					X,Y,Z = tempvector[k]
					outfile.write('%11.8e'%X+' '+'%11.8e'%Y+' '+'%11.8e'%Z+'\n')
				outfile.write("</DataArray>"+'\n')
			#</DataArray>
			
			#Scalar: Equivalent plastic strain, <DataArray>
			for scalarkey in saclarlist:
				tempscalar = scalar_dict[scalarkey]
				outfile.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[scalarkey]+'"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
				for i in newnodeamount_N:
					k = newnode[i]
					X = tempscalar[k]
					outfile.write('%11.8e'%X+'\n')
				outfile.write('</DataArray>'+'\n')
			#</DataArray>
			
			#Invariant in Tensor: Stress Mises, Stress Max.Principal, Stress Mid.Principal, Stress Min.Principal,
			#Stress Min.Principal, Stress Pressure, Stress Tresca,Stress Third_Invariant,Logarithmic_strain_Max_Principal
			#Logarithmic strain Min.Principal, Plastic strain Max.Principal, Plastic strain Min.Principal<DataArray>
			for tensorkey in tensorlist:
				invariant = tensor_dict[tensorkey]['invariant']
				i_validinvariant = 0
				for invariantkey in tensor_dict[tensorkey]['validinvariant']:
					outfile.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[tensorkey]+VARIABLEDICT[invariantkey]+'"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
					for i in newnodeamount_N:
						k = newnode[i]
						X = invariant[k][i_validinvariant]
						outfile.write('%11.8e'%X+'\n')
					outfile.write('</DataArray>'+'\n')
					i_validinvariant = i_validinvariant + 1
			#</DataArray>
			outfile.write("</PointData>"+'\n')
			#</PointData>
			
			
			#<Cells> Write cells into vtk files
			outfile.write('<Cells>'+'\n')
			#Connectivity
			outfile.write('<DataArray type="Int32" Name="connectivity" format="ascii">'+'\n')
			if (config['mesh_type'] == 12):
				for i in range(len(newelement)/8):
					outfile.write(str(newelement[i*8])+' '+str(newelement[i*8+1])+' '+str(newelement[i*8+2])+' '+str(newelement[i*8+3])+' '+str(newelement[i*8+4])+' '+str(newelement[i*8+5])+' '+str(newelement[i*8+6])+' '+str(newelement[i*8+7])+'\n')
			if (config['mesh_type'] == 10):
				for i in range(len(newelement)/4):
					outfile.write(str(newelement[i*4])+' '+str(newelement[i*4+1])+' '+str(newelement[i*4+2])+' '+str(newelement[i*4+3])+'\n')
			outfile.write('</DataArray>'+'\n')
			#Offsets
			outfile.write('<DataArray type="Int32" Name="offsets" format="ascii">'+'\n')
			for i in range(len(newelement)/config['mesh_conner']):
				outfile.write(str(i*config['mesh_conner']+config['mesh_conner'])+'\n')
			outfile.write('</DataArray>'+'\n')
			#Type
			outfile.write('<DataArray type="UInt8" Name="types" format="ascii">'+'\n')
			for i in range(len(newelement)/config['mesh_conner']):
				outfile.write(str(config['mesh_type'])+'\n')
			outfile.write('</DataArray>'+'\n')
			outfile.write('</Cells>'+'\n')
			#</Cells>
			
			
			#</Piece>
			outfile.write('</Piece>'+'\n')
			#</UnstructuredGrid>
			outfile.write('</UnstructuredGrid>'+'\n')
			#</VTKFile>
			outfile.write('</VTKFile>'+'\n')
		
			outfile.close()
			print "Time elapsed: ", time() - time1, "s"	
		
		#====================================写入.pvtu文件=======================================#
		print "Creating .pvtu file for frame ", i_frame," ......"
		time1 = time()
		#create .pvtu files for parallel visualization
		if ( config['piece_number'] > 1 ):
			outfile = open (os.path.join(config['vtk_path'],config['odbname'])+'_'+stepname+'_'+'f%03d'%int(i_frame)+'.pvtu','w')
			
			#write the basic information for .pvtu files
			outfile.write('<?xml version="1.0"?>'+'\n')
			outfile.write('<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian">'+'\n')
			outfile.write("<PUnstructuredGrid GhostLevel="+'"'+str(config['piece_number'])+'"'+">"+'\n')
			#pointdata
			
			#<PointData> Write results data into vtk files
			templist = []
			templist.append("<PPointData ")
			templist.append("Tensors="+'"')
			for tensorkey in tensorlist:
				templist.append(VARIABLEDICT[tensorkey])
				templist.append(',')
			templist.pop()
			templist.append('"'+" ")
			templist.append("Vevtors="+'"')
			for vectorkey in vectorlist:
				templist.append(VARIABLEDICT[vectorkey])
				templist.append(',')
			templist.pop()
			templist.append('"'+" ")
			templist.append("Scalars="+'"')
			for scalarkey in saclarlist:
				templist.append(VARIABLEDICT[scalarkey])
				templist.append(',')
			for tensorkey in tensorlist:
				for invariantkey in tensor_dict[tensorkey]['validinvariant']:
					templist.append(VARIABLEDICT[tensorkey]+VARIABLEDICT[invariantkey])
					templist.append(',')
			templist.append('"'+">"+'\n')
			outfile.write("".join(templist))
								
			for tensorkey in tensorlist:
				outfile.write("<"+"PDataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[tensorkey]+'"'+" "+"NumberOfComponents="+'"'+"9"+'"'+" "+"/>"+'\n')
			for vectorkey in vectorlist:
				outfile.write("<"+"PDataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[vectorkey]+'"'+" "+"NumberOfComponents="+'"'+"3"+'"'+" "+"/>"+'\n')
			for scalarkey in saclarlist:
				outfile.write("<"+"PDataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[scalarkey]+'"'+" "+"/>"+'\n')
			for tensorkey in tensorlist:
				for invariantkey in tensor_dict[tensorkey]['validinvariant']:	
					outfile.write("<"+"PDataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[tensorkey]+VARIABLEDICT[invariantkey]+'"'+" "+"/>"+'\n')
			outfile.write("</PPointData>"+'\n')
			
			#points
			outfile.write("<PPoints>"+'\n')
			outfile.write("<PDataArray type="+'"'+"Float64"+'"'+" "+"NumberOfComponents="+'"'+"3"+'"'+"/>"+'\n')
			outfile.write("</PPoints>"+'\n')
			
			#write the path of each piece for reading it through the .pvtu file 
			for pn in piecekey:
				outfile.write("<Piece Source="+'"'+config['odbname']+'_'+stepname+'_'+instancename+'f%03d'%int(i_frame)+' '+'p'+str(pn)+'.vtu'+'"'+"/>"+'\n')
			
			outfile.write("</PUnstructuredGrid>"+'\n')
			outfile.write("</VTKFile>")

			outfile.close()	
		print "Time elapsed: ", time() - time1, "s"	


#Read attribution from each frame and write to VTK file format in parallel
def WriteToVTKParallel(config = {}, newmodel = {}, node_coordination = [], stepname = '', instancename = '', max_nodelabel = 0):
	frame_list = config['input_frame']
	frame_number = len(frame_list)
	process_munber = config['frame_processing_number']
	frame_each_process = frame_number/process_munber
	control_dict = {}
	#distribute pieces to process
	for i in range(process_munber-1):
		control_dict[str(i)] = frame_list[i*frame_each_process:(i+1)*frame_each_process]
	control_dict[str(process_munber-1)]	= frame_list[(process_munber-1)*frame_each_process : frame_number]
	print "Reorganize by multiprocessing......"
	processlist = []
	for i in range(process_munber):
		P = multiprocessing.Process(target = WriteToVTKSubFunction,\
		args = (config, newmodel, node_coordination, stepname, instancename, control_dict[str(i)], max_nodelabel))
		processlist.append(P)
		P.start()
		print "process-ID: ",P.pid

	for i in processlist:
		P.join()
	
	for i in processlist:
		P.terminate()	

def WriteToVTKSubFunction(config, newmodel, node_coordination, stepname, instancename, framelist, max_nodelabel):
	#open an ODB ( Abaqus output database )
	odb = openOdb(os.path.join(config['odb_path'],config['odbname'])+'.odb',readOnly=True)
	element = odb.rootAssembly.instances[instancename].elements
	frame = odb.steps[stepname].frames
	piecenum = config['piece_number']
	#print("ODB opened!")
	
	print framelist
	for i_frame in framelist:
		#Detect whether the input frame is out of range
		try:
			TRY = frame[int(i_frame)]
		except:
			print "input frame exceeds the range of frames" 
			os._exit(0)
		
		#Access a frame
		n_frame = frame[int(i_frame)]
		print "Frame:",i_frame
				
		#Access Vector: Spatial displacement, acceleration, velocity and reaction force
		vectorlist = config['output_vector']
		print "Reading"+str(vectorlist)+" ......"
		#time1 = time()
		vector_amount = len(vectorlist)
		vector_dict = dict.fromkeys(vectorlist)
		for vector_name in vectorlist:
			tempvector = ReadVector(n_frame.fieldOutputs[vector_name],max_nodelabel)
			vector_dict[vector_name] = copy.deepcopy(tempvector)
		#print "Time elapsed: ", time() - time1, "s"
		
		#modification
		#Access Tensor: Stress components, Logarithmic strain components and Plastic strain components
		tensorlist = config['output_tensor']
		tensor_dict = dict.fromkeys(tensorlist)
		print "Reading"+str(tensorlist)+"......"
		#time1 = time()
		#access Stress components
		tensor_dict[tensorlist[0]] = ReadTensorAtIntS(n_frame.fieldOutputs[tensorlist[0]],7,max_nodelabel)
		#Logarithmic strain components
		tensor_dict[tensorlist[1]] = ReadTensorAtIntE(n_frame.fieldOutputs[tensorlist[1]],2,max_nodelabel)
		#Plastic strain components
		tensor_dict[tensorlist[2]] = ReadTensorAtIntE(n_frame.fieldOutputs[tensorlist[2]],2,max_nodelabel)
		#print "Time elapsed: ", time() - time1, "s"
		
		#Access Saclar: Equivalent plastic strain
		saclarlist = config['output_scalar']
		scalar_dict = dict.fromkeys(saclarlist)
		print "Reading"+str(saclarlist)+"......"
		#Equivalent plastic strain
		#time1 = time()
		scalar_dict[saclarlist[0]] = ReadScalarAtInt(n_frame.fieldOutputs[saclarlist[0]],max_nodelabel)
		#print "Time elapsed: ", time() - time1, "s"	
			
		#==================================以下为写入.vtk文件==================================#			
		piecekey = newmodel.keys()
		for pn in piecekey:
			print "Create .vtu file for piece " + pn + "......"
			#create and open a VTK(.vtu) files
			if(piecenum > 1):
				outfile = open (os.path.join(config['vtk_path'],config['odbname'])+'_'+stepname+'_'+instancename+'f%03d'%int(i_frame)+' '+'p'+str(pn)+'.vtu','w')
			if(piecenum == 1):
				outfile = open (os.path.join(config['vtk_path'],config['odbname'])+'_'+stepname+'_'+instancename+'f%03d'%int(i_frame)+'.vtu','w')
			
			#<VTKFile>, including the type of mesh, version, and byte_order
			outfile.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">'+'\n')
			#<UnstructuredGrid>
			outfile.write('<UnstructuredGrid>'+'\n')
			
			
			#get the partitioned model
			newnode = newmodel[pn]['node']
			newelement = newmodel[pn]['element']
			#compute point & element quantity
			newelementamount = newelement.shape[0] / config['mesh_conner']
			newnodeamount = newnode.shape[0]
			newnodeamount_N = range(0,newnodeamount)
			#<Piece>, including the number of points and cells
			outfile.write('<Piece NumberOfPoints="'+str(newnodeamount)+'"'+' '+'NumberOfCells="'+str(newelementamount)+'">'+'\n')
			
			
			#<Points> Write nodes into vtk files
			tempvector = vector_dict['U']
			outfile.write('<Points>'+'\n')
			outfile.write('<DataArray type="Float64" NumberOfComponents="3" format="ascii">'+'\n')
			for i in newnodeamount_N:
				k = newnode[i]
				X,Y,Z = node_coordination[k]+tempvector[k]
				outfile.write(' '+'%11.8e'%X+'  '+'%11.8e'%Y+'  '+'%11.8e'%Z+'\n')			
			outfile.write('</DataArray>'+'\n')
			outfile.write('</Points>'+'\n')
			#</Points>
			
			
			#<PointData> Write results data into vtk files
			templist = []
			templist.append("<PointData ")
			templist.append("Tensors="+'"')
			for tensorkey in tensorlist:
				templist.append(VARIABLEDICT[tensorkey])
				templist.append(',')
			templist.pop()
			templist.append('"'+" ")
			templist.append("Vevtors="+'"')
			for vectorkey in vectorlist:
				templist.append(VARIABLEDICT[vectorkey])
				templist.append(',')
			templist.pop()
			templist.append('"'+" ")
			templist.append("Scalars="+'"')
			for scalarkey in saclarlist:
				templist.append(VARIABLEDICT[scalarkey])
				templist.append(',')
			for tensorkey in tensorlist:
				for invariantkey in tensor_dict[tensorkey]['validinvariant']:
					templist.append(VARIABLEDICT[tensorkey]+VARIABLEDICT[invariantkey])
					templist.append(',')
			templist.append('"'+">"+'\n')
			outfile.write("".join(templist))
			
			#Tensor: Stress components, Logarithmic strain components, Plastic strain components <DataArray>
			for tensorkey in tensorlist:
				temptensor = tensor_dict[tensorkey]['tensor']
				outfile.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[tensorkey]+'"'+" "+"NumberOfComponents="+'"'+"9"+'"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
				for i in newnodeamount_N:
					k = newnode[i]
					#XX,XY,XZ,YY,YZ,ZZ = temptensor[k][0],temptensor[k][3],temptensor[k][5],temptensor[k][1],temptensor[k][4],temptensor[k][2]
					XX,YY,ZZ,XY,YZ,XZ = temptensor[k]
					YX,ZY,ZX = temptensor[k][3], temptensor[k][4], temptensor[k][5]
					outfile.write('%11.8e'%XX+' '+'%11.8e'%XY+' '+'%11.8e'%XZ+' '+'%11.8e'%YX+' '+'%11.8e'%YY+' '+'%11.8e'%YZ+' '+'%11.8e'%ZX+' '+'%11.8e'%ZY+' '+'%11.8e'%ZZ+'\n')
				outfile.write("</DataArray>"+'\n')
			#</DataArray>
			
			#Vector: Spatial displacement, acceleration, velocity, Reaction force <DataArray>
			for vectorkey in vectorlist:
				tempvector = vector_dict[vectorkey]
				outfile.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[vectorkey]+'"'+" "+"NumberOfComponents="+'"'+"3"+'"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
				for i in newnodeamount_N:
					k = newnode[i]
					X,Y,Z = tempvector[k]
					outfile.write('%11.8e'%X+' '+'%11.8e'%Y+' '+'%11.8e'%Z+'\n')
				outfile.write("</DataArray>"+'\n')
			#</DataArray>
			
			#Scalar: Equivalent plastic strain, <DataArray>
			for scalarkey in saclarlist:
				tempscalar = scalar_dict[scalarkey]
				outfile.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[scalarkey]+'"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
				for i in newnodeamount_N:
					k = newnode[i]
					X = tempscalar[k]
					outfile.write('%11.8e'%X+'\n')
				outfile.write('</DataArray>'+'\n')
			#</DataArray>
			
			#Invariant in Tensor: Stress Mises, Stress Max.Principal, Stress Mid.Principal, Stress Min.Principal,
			#Stress Min.Principal, Stress Pressure, Stress Tresca,Stress Third_Invariant,Logarithmic_strain_Max_Principal
			#Logarithmic strain Min.Principal, Plastic strain Max.Principal, Plastic strain Min.Principal<DataArray>
			for tensorkey in tensorlist:
				invariant = tensor_dict[tensorkey]['invariant']
				i_validinvariant = 0
				for invariantkey in tensor_dict[tensorkey]['validinvariant']:
					outfile.write("<"+"DataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[tensorkey]+VARIABLEDICT[invariantkey]+'"'+" "+"format="+'"'+"ascii"+'"'+">"+'\n')
					for i in newnodeamount_N:
						k = newnode[i]
						X = invariant[k][i_validinvariant]
						outfile.write('%11.8e'%X+'\n')
					outfile.write('</DataArray>'+'\n')
					i_validinvariant = i_validinvariant + 1
			#</DataArray>
			outfile.write("</PointData>"+'\n')
			#</PointData>
			
			
			#<Cells> Write cells into vtk files
			outfile.write('<Cells>'+'\n')
			#Connectivity
			outfile.write('<DataArray type="Int32" Name="connectivity" format="ascii">'+'\n')
			if (config['mesh_type'] == 12):
				for i in range(len(newelement)/8):
					outfile.write(str(newelement[i*8])+' '+str(newelement[i*8+1])+' '+str(newelement[i*8+2])+' '+str(newelement[i*8+3])+' '+str(newelement[i*8+4])+' '+str(newelement[i*8+5])+' '+str(newelement[i*8+6])+' '+str(newelement[i*8+7])+'\n')
			if (config['mesh_type'] == 10):
				for i in range(len(newelement)/4):
					outfile.write(str(newelement[i*4])+' '+str(newelement[i*4+1])+' '+str(newelement[i*4+2])+' '+str(newelement[i*4+3])+'\n')
			outfile.write('</DataArray>'+'\n')
			#Offsets
			outfile.write('<DataArray type="Int32" Name="offsets" format="ascii">'+'\n')
			for i in range(len(newelement)/config['mesh_conner']):
				outfile.write(str(i*config['mesh_conner']+config['mesh_conner'])+'\n')
			outfile.write('</DataArray>'+'\n')
			#Type
			outfile.write('<DataArray type="UInt8" Name="types" format="ascii">'+'\n')
			for i in range(len(newelement)/config['mesh_conner']):
				outfile.write(str(config['mesh_type'])+'\n')
			outfile.write('</DataArray>'+'\n')
			outfile.write('</Cells>'+'\n')
			#</Cells>
			
			
			#</Piece>
			outfile.write('</Piece>'+'\n')
			#</UnstructuredGrid>
			outfile.write('</UnstructuredGrid>'+'\n')
			#</VTKFile>
			outfile.write('</VTKFile>'+'\n')
		
			outfile.close()
			#print "Time elapsed: ", time() - time1, "s"	
		
		#====================================写入.pvtu文件=======================================#
		print "Creating .pvtu file for frame ", i_frame," ......"
		time1 = time()
		#create .pvtu files for parallel visualization
		if ( piecenum > 1 ):
			outfile = open (os.path.join(config['vtk_path'],config['odbname'])+'_'+stepname+'_'+'f%03d'%int(i_frame)+'.pvtu','w')
			
			#write the basic information for .pvtu files
			outfile.write('<?xml version="1.0"?>'+'\n')
			outfile.write('<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian">'+'\n')
			outfile.write("<PUnstructuredGrid GhostLevel="+'"'+str(piecenum)+'"'+">"+'\n')
			#pointdata
			
			#<PointData> Write results data into vtk files
			templist = []
			templist.append("<PPointData ")
			templist.append("Tensors="+'"')
			for tensorkey in tensorlist:
				templist.append(VARIABLEDICT[tensorkey])
				templist.append(',')
			templist.pop()
			templist.append('"'+" ")
			templist.append("Vevtors="+'"')
			for vectorkey in vectorlist:
				templist.append(VARIABLEDICT[vectorkey])
				templist.append(',')
			templist.pop()
			templist.append('"'+" ")
			templist.append("Scalars="+'"')
			for scalarkey in saclarlist:
				templist.append(VARIABLEDICT[scalarkey])
				templist.append(',')
			for tensorkey in tensorlist:
				for invariantkey in tensor_dict[tensorkey]['validinvariant']:
					templist.append(VARIABLEDICT[tensorkey]+VARIABLEDICT[invariantkey])
					templist.append(',')
			templist.append('"'+">"+'\n')
			outfile.write("".join(templist))
								
			for tensorkey in tensorlist:
				outfile.write("<"+"PDataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[tensorkey]+'"'+" "+"NumberOfComponents="+'"'+"9"+'"'+" "+"/>"+'\n')
			for vectorkey in vectorlist:
				outfile.write("<"+"PDataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[vectorkey]+'"'+" "+"NumberOfComponents="+'"'+"3"+'"'+" "+"/>"+'\n')
			for scalarkey in saclarlist:
				outfile.write("<"+"PDataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[scalarkey]+'"'+" "+"/>"+'\n')
			for tensorkey in tensorlist:
				for invariantkey in tensor_dict[tensorkey]['validinvariant']:	
					outfile.write("<"+"PDataArray"+" "+"type="+'"'+"Float32"+'"'+" "+"Name="+'"'+VARIABLEDICT[tensorkey]+VARIABLEDICT[invariantkey]+'"'+" "+"/>"+'\n')
			outfile.write("</PPointData>"+'\n')
			
			#points
			outfile.write("<PPoints>"+'\n')
			outfile.write("<PDataArray type="+'"'+"Float64"+'"'+" "+"NumberOfComponents="+'"'+"3"+'"'+"/>"+'\n')
			outfile.write("</PPoints>"+'\n')
			
			#write the path of each piece for reading it through the .pvtu file 
			for pn in piecekey:
				outfile.write("<Piece Source="+'"'+config['odbname']+'_'+stepname+'_'+instancename+'f%03d'%int(i_frame)+' '+'p'+str(pn)+'.vtu'+'"'+"/>"+'\n')
			
			outfile.write("</PUnstructuredGrid>"+'\n')
			outfile.write("</VTKFile>")
			outfile.close()


#---------------------Read Attribution and Output to VTK--------------------#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
###==============================Sub=Function=============================###


###===========================Convertion Function=========================###
#Convertion Function
#Modify the default value of filename here to specify the default configuration file
def ConvertOdb2Vtk(filename = 'ODB2VTK+.cfg'):
	
	starttime = time()
	#read the configuration file, return a dictionary storing parameters
	config = GetConfig(filename)
	#open an ODB ( Abaqus output database )
	odb = openOdb(os.path.join(config['odb_path'],config['odbname'])+'.odb',readOnly=True)
	print "ODB opened!"

	#access geometry and topology information ( odb->rootAssembly->instances->(nodes, elements) )
	rootassembly = odb.rootAssembly
	instance = rootassembly.instances
	#access attribute information ( odb->steps->frames->fieldOutputs )
	step = odb.steps
	#get instance & step information : Quantity and all names
	instance_key = instance.keys()
	instance_amount = len(instance_key)
	step_key = step.keys()
	step_amount = len(step_key)
	#check whether the input is out of range of instances or steps
	for i in config['input_step']:
		if(step_amount < int(i) or step_amount == int(i)):
			print "input step exceeds the range of steps"
			os._exit(0)
	for i in config['input_instance']:
		if(instance_amount < int(i) or step_amount == int(i)):
			print "input instance exceeds the range of instances"
			os._exit(0)
	
	#instance cycle
	for i_instance in config['input_instance']:
		instancename = instance_key[int(i_instance)]
		print "Instance: ",instancename
		#access nodes & elements
		node = instance[instancename].nodes
		element = instance[instancename].elements
		node_amount = len(node)
		element_amount = len(element)
		
		#get the max label of node, the max label is usually not equal to the amount of node
		#because there might have some empty nodes in mesh
		temptime = time()
		max_nodelabel = node[node_amount-1].label
		for i_node in node:
			if(i_node.label > max_nodelabel):
				max_nodelabel = i_node.label
		print "max nodelabel got, time elapsed: ", time() - temptime
		
		#get all nodes' coordination
		temptime = time()
		node_coordination = ReadNode(node,max_nodelabel)
		print "node coordination read, time elapsed: ", time() - temptime
		
		#===================================================================================#
		#===PARTITION&REORGANIZATION: partition the single instance and Reorganize pieces===#
		
		#Partition and reorganization the model in parallel using multiprocessing
		if(config['parallel_partition'] == 'TRUE'):
			#Partitioned: Partition instance compute the number of element of each block
			print "Partitionning model into pieces ......"
			temptime = time()
			if(config['partition_method'] == 'Random'):
				piecedict = PartitionRandom(config, element_amount)
			elif(config['partition_method'] == 'Regular'):
				piecedict, element_connectivity = PartitionRegularParallel(config, node_coordination, instancename)	
			elif(config['partition_method'] == 'CenterPoint'):
				piecedict, element_connectivity = PartitionPointCenterParallel(config, node_coordination, instancename)	
			else:
				print "partition method" + config['partition_method'] + "doesn't exist, it must be 'Random','Regular' or 'CenterPoint'" 
				os._exit(0)
			print "Partition Time elapsed: ", time() - temptime, "s"
			
			#Reorganization: Reorganize node and element
			print "Reorganize node and element to model ......"
			temptime = time()
			if(config['partition_method'] == 'Random'):
				newmodel = ReorganizeRandomNodeElementParallel(config, piecedict, instancename, max_nodelabel)
			if(config['partition_method'] == 'Regular' or config['partition_method'] == 'CenterPoint'):
				newmodel = ReorganizeNodeElementParallel(config, piecedict, element_connectivity, max_nodelabel)
			print "Reorganization Time elapsed: ", time() - temptime, "s"
		
		#Partition and reorganization the model in serial
		elif(config['parallel_partition'] == 'FALSE'):
			#Partitioned: Partition instance compute the number of element of each block
			print "Partitionning model into pieces ......"
			temptime = time()
			if(config['partition_method'] == 'Random'):
				piecedict = PartitionRandom(config, element_amount)
			elif(config['partition_method'] == 'Regular'):
				piecedict, element_connectivity = PartitionRegularSerial(config, node_coordination, element)	
			elif(config['partition_method'] == 'CenterPoint'):
				piecedict, element_connectivity = PartitionPointCenterSerial(config, node_coordination, element)	
			else:
				print "Partition method error: ", config['partition_method'], " is illegal! It must be 'Random','Regular' or 'CenterPoint'" 
				os._exit(0)
			print "Partition Time elapsed: ", time() - temptime, "s"
			
			#Reorganization: Reorganize node and element
			print "Reorganize node and element to model ......"
			temptime = time()
			if(config['partition_method'] == 'Random'):
				newmodel = ReorganizeRandomNodeElementSerial(config, piecedict, element, max_nodelabel)
			if(config['partition_method'] == 'Regular' or config['partition_method'] == 'CenterPoint'):
				newmodel = ReorganizeNodeElementSerial(config, piecedict, element_connectivity, max_nodelabel)
			print "Reorganization Time elapsed: ", time() - temptime, "s"
		
		#Error: Parameter error
		else:
			print "Parameter error: ", config['parallel_partition'], " is illegal! It must be 'True' or 'FALSE'"
			os._exit(0)
		
		#========================PARTITION&REORGANIZATION FINISHED!=========================#
		
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
		
		#===================================================================================#
		#step cycle
		for i_step in config['input_step']:
			stepname = step_key[int(i_step)]
			print "Step: ",stepname
			#access attribute(fieldOutputs) information
			frame = step[stepname].frames
			if(config['parallel_frame'] == 'TRUE'):
				WriteToVTKParallel(config, newmodel, node_coordination, stepname, instancename, max_nodelabel)
			elif(config['parallel_frame'] == 'FALSE'):
				WriteToVTKSerial(config, frame, newmodel, node_coordination, stepname, instancename, max_nodelabel)
			else:
				print "Parameter error: ", config['parallel_frame'], " is illegal! It must be 'True' or 'FALSE'"
				os._exit(0)
	
	odb.close()
	print "Convertion time elapsed: ", time() - starttime, "s"


if __name__ == '__main__':
    #input the configuration file path
	ConvertOdb2Vtk(filename = r'ODB2VTK+.cfg')
