# Reproducible model evaluation

The evaluation workflow streams 70 examples from the Food-101 validation split
(research/non-commercial use as approved for this project), deterministically
assigning 20 examples to tuning and 50 to final evaluation. Images and reports
are temporary CI artifacts and are never committed.
