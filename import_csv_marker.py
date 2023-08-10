import Metashape, os

def import_markers_csv():
	doc = Metashape.app.document
	if not (len(doc.chunks)):
		print("No chunks. Script aborted.\n")
		return False
	
	path = Metashape.app.getOpenFileName("Select markers import file:")
	if path == "":
		print("Incorrect path. Script aborted.\n")
		return False
	print("Markers import started.\n")  #informational message
	file = open(path, "rt")	#input file

	# Skip header line
	file.readline()
	eof = False
 
	line = file.readline()
	if not len(line):
		eof = True

	chunk = doc.chunk

	while not eof:	
		sp_line = line.strip().rsplit(",", 3)   #splitting read line
		y = float(sp_line[3])			#x- coordinate of the current projection in pixels
		x = float(sp_line[2])			#y- coordinate of the current projection in pixels
		label = sp_line[0]				#image file name
		marker_name = sp_line[1]		#marker label
		flag = 0
        
		for camera in chunk.cameras:

			if os.path.basename(camera.photo.path).lower() == label.lower():		#searching for the camera
	
				for marker in chunk.markers:	#searching for the marker (comparing with all the marker labels in chunk)
					if marker.label.lower() == marker_name.lower():
						marker.projections[camera] = Metashape.Marker.Projection(Metashape.Vector([x, y]), True)		#setting up marker projection of the correct photo)
						flag = 1
						break

				if not flag:   #creating new marker instance
					marker = chunk.addMarker() 
					marker.label = marker_name
					marker.projections[camera] = Metashape.Marker.Projection(Metashape.Vector([x, y]), True)
										
				break
				
		line = file.readline()		#reading the line from input file
		if not len(line):
			eof = True
			break # EOF
	file.close()	
    
	print("Markers import finished.\n")
	return True

Metashape.app.addMenuItem("Custom menu/Import markers from CSV file", import_markers_csv)
