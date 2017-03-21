# brain is a machine learning library for go

## Warning

brain is a personal experimental project that under heavy development, backwark api compatibility is not guaranteed!

## Examples

### KNN

```go
data, target := datasets.LoadIris()
clf := knn.KNeighborsClassifier(5)
clf.Fit(data, target)
log.Println(clf.Predict([]float64{5.9, 3, 5.1, 1.8}))
```

### Navie bayes

```go
listOfPosts, listClasses := datasets.LoadSpam()
vocabList := datasets.CreateVocabList(listOfPosts)
inputVec := datasets.SetOfWords2Vec(vocabList, listOfPosts[2])
nb := BernoulliNB()

var trainMat [][]float64
for _, v := range listOfPosts {
    trainMat = append(trainMat, datasets.SetOfWords2Vec(vocabList, v))
}

nb.Fit(trainMat, listClasses)
t.Log(nb.Predict(inputVec))

t.Log(nb.Measure(trainMat, listClasses))
```

## Todo

- [x] knn
- [x] naive bayes
- [ ] svm
- [ ] tree
- [ ] neural network
