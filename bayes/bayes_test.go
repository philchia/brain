package bayes

import "testing"

func TestBernoulliNB(t *testing.T) {
	listOfPosts, listClasses := loadSpam()
	vocabList := createVocabList(listOfPosts)
	inputVec := setOfWords2Vec(vocabList, listOfPosts[2])
	nb := BernoulliNB()

	var trainMat [][]float64
	for _, v := range listOfPosts {
		trainMat = append(trainMat, setOfWords2Vec(vocabList, v))
	}
	t.Log(trainMat)
	nb.Fit(trainMat, listClasses)
	t.Log(nb.Predict(inputVec))
	t.Log(listClasses[2])
	t.Log(nb.Measure(trainMat, listClasses))
}

func loadSpam() (data [][]string, target []string) {
	postingList := [][]string{
		{"my", "dog", "has", "flea", "problems", "help", "please"},
		{"maybe", "not", "take", "him", "to", "dog", "park", "stupid"},
		{"my", "dalmation", "is", "so", "cute", "I", "love", "him"},
		{"stop", "posting", "stupid", "worthless", "garbage"},
		{"mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"},
		{"quit", "buying", "worthless", "dog", "food", "stupid"},
	}

	classVec := []string{"nospam", "spam", "nospam", "spam", "nospam", "spam"}
	return postingList, classVec
}

func createVocabList(data [][]string) []string {
	vocabSet := []string{}

	for _, post := range data {
	TOKENLOOP:
		for _, token := range post {
			for _, vacab := range vocabSet {
				if vacab == token {
					continue TOKENLOOP
				}
			}
			vocabSet = append(vocabSet, token)
		}
	}
	return vocabSet
}

func setOfWords2Vec(vocabList []string, input []string) []float64 {
	retVec := make([]float64, len(vocabList))

	for _, word := range input {
		for i, vocab := range vocabList {
			if word == vocab {
				retVec[i] = 1
			}
		}
	}
	return retVec
}
