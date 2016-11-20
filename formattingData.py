import csv
import os

"""
guns: 2  dem
abortion: 1 = dem
creation: 2 = dem
gayRights: 1 = dem
God: 2  =dem
healthcare: 1 = dem
"""
def saveText():
 	folder = '/Users/gregmiller/Desktop/SomasundaranWiebe-politicalDebates/'
 	topic = 'guns'
 	abortion = folder + 'abortion'
 	creation = folder  + 'creation'
 	gayRights = folder + 'gayRights'
 	god = folder +  'god'
 	guns = folder  + 'guns'
 	healthcare = folder + 'healthcare'
 	path  = guns
 	with open('formatted' + topic + '.csv', 'a') as smallFile:
	 	with open('formattedDebate.csv', 'a') as csvFile:
	 		for filename in os.listdir(path):
	 			with open(path + '/'+ filename, 'r') as f:
	 				count = 0
	 				data = ''
	 				for line in f :
	 					count += 1 
	 					if count == 1:
							stance = int(line[len('#stance=stance')])
							if topic == 'creation' or topic == 'god' or topic == 'guns':
								if stance == 2:
									stance = 1
								elif stance ==1:
									stance = 2
						if count > 3:
							data = data + line
					fields = [stance, data]
					writer = csv.writer(csvFile)
					newWriter  = csv.writer(smallFile)

					writer.writerow(fields)
					newWriter.writerow(fields)



saveText()
