package bayes

import (
	"log"
	"math"

	"github.com/philchia/brain"
)

// MultinomialNB create a new multinomial naive bayes classifier
func MultinomialNB() brain.Classifier {
	return nil
}

// BernoulliNB create a new bernoulli navie bayes classifier
func BernoulliNB() brain.Classifier {
	return &bayes{}
}

// GaussianNB create a gaussian naive bayes classifier
func GaussianNB() brain.Classifier {
	return nil
}

type bayes struct {
	tokenPB map[string][]float64
	catePB  map[string]float64
}

func (b *bayes) Fit(x [][]float64, y []string) {

	categories := map[string]int{}
	for _, category := range y {
		categories[category]++
	}

	numTrainDocs := len(x)

	pb := map[string]float64{}
	for k, v := range categories {
		pb[k] = float64(v) / float64(numTrainDocs)
	}

	numWords := len(x[0])

	vpb := map[string][]float64{}
	vpc := map[string]int{}
	for k := range categories {
		vpc[k] = 2
		vpb[k] = onesF(numWords)
	}

	for i := 0; i < numTrainDocs; i++ {
		cate := y[i]
		if _, ok := vpb[cate]; !ok {
			vpb[cate] = make([]float64, numWords)
		}
		sumVF(vpb[cate], x[i])
		vpc[cate] += int(sumF(x[i]))
	}

	for k, v := range vpb {
		multipleF(v, 1/float64(vpc[k]))
	}

	for k, v := range vpb {
		for i, f := range v {
			vpb[k][i] = logF(f)
		}
	}

	b.catePB = pb
	b.tokenPB = vpb
}

func (b *bayes) Predict(x []float64) string {
	var ret string
	var maxPB float64
	for k, v := range b.catePB {
		pb := sumF(multipleVFS(x, b.tokenPB[k])) + logF(v)

		log.Println(pb, k)
		if pb > maxPB || maxPB == 0 {
			maxPB = pb
			ret = k
		}
	}
	return ret
}

func (b *bayes) Measure(x [][]float64, y []string) float64 {
	total := len(y)
	res := 0
	for i, target := range y {
		if target == b.Predict(x[i]) {
			res++
		}
	}
	return float64(res) / float64(total)
}

func onesI(l int) []int {
	ret := make([]int, l)
	for i := range ret {
		ret[i] = 1
	}
	return ret
}

func onesF(l int) []float64 {
	ret := make([]float64, l)
	for i := range ret {
		ret[i] = 1
	}
	return ret
}

func sumVF(v []float64, s []float64) {

	for i, n := range s {
		v[i] += n
	}
}

func sumI(v []int) int {
	ret := 0
	for _, n := range v {
		ret += n
	}
	return ret
}

func sumF(v []float64) float64 {
	ret := 0.0
	for _, n := range v {
		ret += n
	}
	return ret
}

func multipleN(v []int, s int) {
	for i, n := range v {
		v[i] = n * s
	}
}

func multipleF(v []float64, s float64) {
	for i, n := range v {
		v[i] = n * s
	}
}

func multipleVN(v []int, s []int) {
	for i, n := range v {
		v[i] = n * s[i]
	}
}

func multipleVF(v []float64, s []float64) {
	for i, n := range v {
		v[i] = n * s[i]
	}
}

func multipleVFS(v []float64, s []float64) []float64 {
	var ret = make([]float64, len(v))
	for i, n := range v {
		ret[i] = n * s[i]
	}
	return ret
}

func logF(x float64) float64 {
	return math.Log(x)
}

func exp(x float64) float64 {
	return math.Exp(x)
}

func logV(v []float64) {
	for i, n := range v {
		v[i] = logF(n)
	}
}
