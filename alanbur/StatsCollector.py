from recommend import YelpSVM

results=[[]]
percents=[.00000190734, 0.00000381469, 0.00000762939, 0.00001525878, 0.00003051757, \
	0.00006103515,0.00012207031,0.00024414062,0.00048828125,0.0009765625,0.001953125,0.00390625,\
	0.0078125,0.015625,0.03125,0.0625,0.125, .250]
modes=['random','longestReviews', 'shortestReviews', 'percentile','percentile','percentile','percentile' ]
YSVM=YelpSVM()
YSVM.getData()		#read in all data


numTrials=5
entry=[]
percentile=-.25
# for fullIter in range (totalNumTrials):
for i in range(len(modes)):
	currentMode=modes[i]
	results.append([])
	
	if currentMode=='percentile':
		percentile+=.25
	for percent in percents:
		# if currentMode=='longestReviews':
		# 	YSVM.getMostWords(percent)
		# elif currentMode=='shortestReviews':
		# 	YSVM.getLeastWords(percent)
		
		entry=[]
		for j in range(numTrials):
			try:
				YSVM.testData(5000)	#reset test data set
				print(percentile)
				entry.append(YSVM.run(percent, False, currentMode, percentile))
			except:
				entry.append(0)
		results[i].append(entry)
	
print(results)
# YSVM.run(, False)
# YSVM.run(.0000095, False)
# YSVM.run(.0000095, False)
# YSVM.run(.0000095, False)
# YSVM.run(.0000095, False)
# YSVM.run(.0000095, False)

# YSVM.run(.25, False)
# YSVM.run(.125, False)
# YSVM.run(.0625, False)
# YSVM.run(.01, False)
