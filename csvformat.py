import os 
for f in os.listdir("predictions"):
    if ".txt" in f:
		with open("predictions/"+f) as k:
			chars = k.read().split()
		with open("fp/" + f[:-3]+"csv", 'w') as k:
			k.write("ID,Class\n")
			for i,c in enumerate(chars):
				k.write(str(i+1)+","+str(c)+"\n")