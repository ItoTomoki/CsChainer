import numpy as np
import pylab as plt

testdammydata = np.array([[1,1,3,4],[1,2,3],[0,0,0,0,5,5]])
preddamydata = np.array([[1,2,0,4],[1,1,1],[0,0,3,3,5,5]])

#データの読み込み数で比較
ResultTrueFalsearray = [testdammydata[i] == preddamydata[i] for i in range(len(testdammydata))]
Test1 = np.array([testdammydata[i][0] for i in range(3)])
Pred1 = np.array([preddamydata[i][0] for i in range(3)])
print len(Test1[Test1 == Pred1])
Test2 = np.array([testdammydata[i][1] for i in range(3)])
Pred2 = np.array([preddamydata[i][1] for i in range(3)])
print len(Test2[Test2 == Pred2])
Test3 = np.array([testdammydata[i][2] for i in range(3)])
Pred3 = np.array([preddamydata[i][2] for i in range(3)])
print len(Test4[Test4 == Pred3])

Test4 = []
for i in range(len(testdammydata)):
	try:
		Test4.append(testdammydata[i][3])
	except:
		continue


Test4 = np.array([try:testdammydata[i][3] for i in range(3) except: continue])
Pred4 = np.array([preddamydata[i][3] for i in range(3)])
def createConpArray(indexnumber,data):
	Testi = []
	for i in range(len(data)):
		try:
			Testi.append(data[i][indexnumber])
		except:
			continue
	return np.array(Testi)

AccuracyArray = []
for indexnumber in range(5):
	Test = createConpArray(indexnumber,testdammydata)
	Pred = createConpArray(indexnumber,preddamydata)
	print len(Test[Test == Pred])
	if len(Test) != 0:
		Accuracy = float(len(Test[Test == Pred]))/float(len(Test))
		AccuracyArray.append([indexnumber,Accuracy])

plt.xlabel("number of data")
plt.ylabel("accuracy")
plt.plot(np.array(AccuracyArray).T[0],np.array(AccuracyArray).T[1],color = "blue", label = "RNN")
plt.plot(np.array(AccuracyArray).T[0],[0,1,2,3,4],color = "red",label = "SVM")	
plt.xlabel("number")
plt.ylabel("acuracy")

plt.show()
plt.close()

TrueFalsearray = np.array([testdammydata[2][1::][i] == testdammydata[2][0:-1][i] for i in range(len(testdammydata[2])-1)])
g = testdammydata[2]
g2 = preddamydata[2]
sameTestarray = []
defferentaTestrray = []
samePredarray = []
defferentaPredarray = []
for k in range(len(g)-1):
	if g[k] == g[k + 1]:
		sameTestarray.append(g[k])
		samePredarray.append(g2[k])
	else:
		defferentaTestrray.append(g[k])
		defferentaPredarray.append(g2[k])

from sklearn.metrics import accuracy_score
print accuracy_score(np.array(defferentaTestrray),np.array(defferentaPredarray))

def createConpArray2(testdata,preddata):
	Testi = []
	for i,j in range(testdata[1::],testdata[0:-1]):
		try:
			Testi.append(data[i][indexnumber])
		except:
			continue


