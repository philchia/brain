package datasets

// LoadSpam will return sparm token slice and label slice
func LoadSpam() (data [][]string, target []string) {
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

// CreateVocabList return vocab list from input
func CreateVocabList(data [][]string) []string {
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

// SetOfWords2Vec create word vec
func SetOfWords2Vec(vocabList []string, input []string) []float64 {
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
