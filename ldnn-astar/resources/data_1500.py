standard_prompt='''
# High Quality Output Path

# Long straight narrow path (do not choose other waypoints, long straight narrow path is between two long straight parallel obstacles)
Start Point : [8,12]
Goal Point :[25,12]
Horizontal Barriers :  [[13,0,28],[11,0,28]]
Vertical Barriers : []
Waypoints :  [[8,12], [25,14]]
Path : [[8,12],[9,12],[10,12],[11,12],[12,12],[13,12],[14,12],[15,12],[16,12],[17,12],[18,12],[19,12],[20,12],[21,12],[22,12],[23,12],[24,12],[25,12]]

# Multiple-angle path (only choose waypoints around angles, angles are the junctions of horizontal barriers and vertical barriers)
Start Point : [20,18]
Goal Point : [18,4]
Horizontal Barriers : [[16,16,32],[6,16,42]] 
Vertical Barriers : [[16,6,16]]
Waypoints :  [[20, 18], [15, 16], [16, 5], [18, 4]]
Path : [[20, 18], [19, 17], [18, 17], [17, 17], [16, 17], [15, 16], [15, 15], [15, 14], [15, 13], [15, 12], [15, 11], [15, 10], [15, 9], [15, 8], [15, 7], [15, 6], [16, 5], [17, 4], [18, 4]]

# comprehensive situation

Start Point : [30,28]
Goal Point :[36,20]
Horizontal Barriers : [[30,28,32]]
Vertical Barriers : [[24,12,24],[28,18,30],[32,12,30]]
Waypoints :  [[30,28],[31,21],[32,11],[34,16],[36,20]]
Path : [[30, 28], [31, 27], [31, 26], [31, 25], [31, 24], [31, 23], [31, 22], [31, 21], [31, 20], [31, 19], [31, 18], [31, 17], [31, 16], [31, 15], [31, 14], [31, 13], [31, 12], [32, 11], [33, 12], [34, 13], [34, 14], [34, 15], [34, 16], [35, 17], [36, 18], [36, 19], [36, 20]]

Start Point : [10, 12]
Goal Point :[24, 6]
Horizontal Barriers : [[30,28,32]]
Vertical Barriers : [[24,12,24],[28,18,30],[32,12,30]]
Waypoints :  [[10, 12], [12, 12], [24, 6]]
Path : [[30, 28], [31, 27], [31, 26], [31, 25], [31, 24], [31, 23], [31, 22], [31, 21], [31, 20], [31, 19], [31, 18], [31, 17], [31, 16], [31, 15], [31, 14], [31, 13], [31, 12], [32, 11], [33, 12], [34, 13], [34, 14], [34, 15], [34, 16], [35, 17], [36, 18], [36, 19], [36, 20]]


'''
