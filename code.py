import sys
from pomegranate import BayesianNetwork
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt

# learn the structure and get 3 sets of data
def getModelAndData(infile, train_percent=.4):
	allData = pd.read_csv(infile)
	print("CSV Read")
	trainData, testDataFull = train_test_split(allData, train_size=train_percent)
	devData, testData = train_test_split(testDataFull, train_size=0.5)

	# LaPlace smoothing - add 1 fraud, 1 not fraud for every 
	columnVals = []
	for col in allData:
		columnVals.append(allData[col].unique().tolist())

	for inst in itertools.product(*columnVals):
		newPoint = {colName: colValue for colName, colValue in zip(list(allData.columns.values), inst)}
		# print(newPoint)
		trainData.append(newPoint, ignore_index=True)

	# create a copy whose fraud column can be emptied
	devDataCopy = devData.copy()
	testDataCopy = testData.copy()

	# set test data to have empty fraud identity
	devDataCopy['isfraud'] = None
	testDataCopy['isfraud'] = None

	# convert out of pandas
	trainData = trainData.values
	devDataCopy = devDataCopy.values
	testDataCopy = testDataCopy.values

	model = BayesianNetwork.from_samples(trainData)

	return model, trainData, devDataCopy, testDataCopy, devData, testData

# perform inference on the structure with the development data
def doInference(model, data):
	return model.predict(data)

# numpy to CSV
def npToCSV(np_array, data_name):
	df = pd.DataFrame(np_array)
	df.to_csv(data_name + ".csv")

# save the train data, dev data, and test data to the machine
def saveModelAndData(model, trainData, devData, testData, devDataComplete, testDataComplete):
	with open("model.ml", 'w') as file:
		file.write(str(model.structure))
	npToCSV(trainData, 'trainData')
	npToCSV(devData, 'devData')
	npToCSV(testData, 'testData')
	npToCSV(devDataComplete, 'devDataComplete')
	npToCSV(testDataComplete, 'testDataComplete')

def plotModel(model):
	model.plot()
	plt.show()

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise Exception("usage: enter csv, output file, (and train_size: defaults to .4)")
    
    infile = sys.argv[1]

    train_percent = .4
    if len(sys.argv) == 4:
    	train_percent = sys.argv[2]

    model, trainData, devData, testData, devDataComplete, testDataComplete = getModelAndData(infile, train_percent)
    print("Model and data obtained")
    saveModelAndData(model, trainData, devData, testData, devDataComplete, testDataComplete)
    print("Model and data saved")

    devPrediction = doInference(model, devData)
    testPrediction = doInference(model, testData)

    npToCSV(devPrediction, 'devPrediction')
    npToCSV(testPrediction, 'testPrediction')

    plotModel(model)

if __name__ == '__main__':
    main()


