package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type covidTracking []struct {
	Date             int `json:"date"`
	Positive         int `json:"positive"`
	PositiveIncrease int `json:"positiveIncrease"`
}

func convertCovidTracking(jsonData []byte) (ret [][]string) {
	var d covidTracking
	json.Unmarshal(jsonData, &d)
	for _, row := range d {
		cd := fmt.Sprint(row.Date)
		ret = append(ret, []string{
			fmt.Sprintf("%s-%s-%s", cd[:4], cd[4:6], cd[6:]),
			fmt.Sprint(row.PositiveIncrease),
			fmt.Sprint(row.Positive),
		})
	}
	return
}

func writeCovidTracking(jsonData []byte, path string) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	w.Write(strings.Split("Date,New Cases,Total Cases", ","))
	w.WriteAll(convertCovidTracking(jsonData))
}
