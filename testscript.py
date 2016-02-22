import sys
import os


def getTxtFiles(a_dir,a):
        return [name for name in os.listdir(a_dir)
                if name.endswith(".t7") and a in name]


for cap_emb in range(30, 100, 30):
	print ("starting iteration with "+str(cap_emb)+" Cap Embedding")
	command = "th HW2.lua -classifier nn -hiddenUnits 200 -wordEmbeddingSize 50 -epochs 20 -learningrate 0.1 -capEmbeddingSize " + str(cap_emb)
	print (command)
	os.system(command)
	#fls = getTxtFiles("cv","lstm_"+str(nesting_depth)+"_"+str(num_cellstates))
	#print fls
	##now sample the trained model
