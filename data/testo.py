from pybaseball import statcast

# Just get any data.
some_pitches = statcast(start_dt="2022-05-01", end_dt="2022-05-14")

some_pitches.to_parquet("data/test_data.parquet")
