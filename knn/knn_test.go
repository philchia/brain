package knn

import (
	"log"
	"testing"

	"github.com/philchia/brain/datasets"
)

func TestNew(t *testing.T) {
	type args struct {
		k int
	}
	tests := []struct {
		name    string
		args    args
		wantNil bool
	}{
		{
			"case1",
			args{
				3,
			},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := KNeighborsClassifier(tt.args.k); (got == nil) != tt.wantNil {
				t.Errorf("New(), want nil %v, got %v", tt.wantNil, got)
			}
		})
	}
}

func Test_knn_Fit(t *testing.T) {
	type fields struct {
		k      int
		data   [][]float64
		target []string
	}
	type args struct {
		x [][]float64
		y []string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{
			"case1",
			fields{},
			args{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &knn{
				k:      tt.fields.k,
				data:   tt.fields.data,
				target: tt.fields.target,
			}
			c.Fit(tt.args.x, tt.args.y)
		})
	}
}

func Test_knn_Predict(t *testing.T) {
	type fields struct {
		k      int
		data   [][]float64
		target []string
	}
	type args struct {
		x []float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   string
	}{
	// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &knn{
				k:      tt.fields.k,
				data:   tt.fields.data,
				target: tt.fields.target,
			}
			if got := c.Predict(tt.args.x); got != tt.want {
				t.Errorf("knn.Predict() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestKnn(t *testing.T) {
	data, target := datasets.LoadIris()
	clf := KNeighborsClassifier(5)
	clf.Fit(data, target)
	log.Println(clf.Predict([]float64{5.9, 3, 5.1, 1.8}))
}
