package bayes

import "testing"
import "github.com/philchia/brain/datasets"

func TestBernoulliNB(t *testing.T) {
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
}
