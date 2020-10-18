package main

import (
	"encoding/csv"
	"os"
	"strings"
)

func writeCountyData(csvData []byte, path string) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	r := csv.NewReader(strings.NewReader(string(csvData)))
	records, err := r.ReadAll()
	if err != nil {
		panic(err)
	}

	w.Write(strings.Split("Date,County,State,FIPS,Cases,Deaths", ","))
	for _, record := range records {
		if record[2] == "New Jersey" {
			w.Write(record)
		}
	}
}
