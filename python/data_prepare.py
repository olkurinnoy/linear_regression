import numpy as np

#load data
location = '../data/pre_data.csv'
data = np.genfromtxt(location, dtype = int, delimiter=',')
data = data[1:,:]


#creating new features
id_and_score = np.unique(data[:,[0,1]], axis=0)
pos_sent = np.array([])
neg_sent = np.array([])


#summing positive and negative sentiments
for iter in id_and_score[:,0]:
	indexes = np.multiply((data[:,0] == iter),data[:,2])
	pos_sent = np.append(pos_sent, np.sum(indexes==1))[:,None]
	neg_sent = np.append(neg_sent, np.sum(indexes==-1))[:,None]
	

#gathering data -- pos/neg sent -- into input X
X = np.append(pos_sent, neg_sent, axis=1)
Y = id_and_score[:,1][:,None]

X = np.append(X, Y, axis=1)

#save into new file
np.savetxt("../data/data.csv", X.astype(int), fmt='%i', delimiter=",")

