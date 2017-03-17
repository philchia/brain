# brain is a machine learning library for go

## KNN

```go
data, target := datasets.LoadIris()
clf := knn.New(5)
clf.Fit(data, target)
log.Println(clf.Predict([]float64{5.9, 3, 5.1, 1.8}))
```

## Todo

- [x] knn
- [] svm
- [] bayes
- [] tree
- [] neural network
