import pyRAPL
import random

pyRAPL.setup() 

csv_output = pyRAPL.outputs.CSVOutput('result.csv')
@pyRAPL.measureit(output=csv_output)
def fucntion_to_measure():
    # Instructions to be evaluated.
    for i in range(1000):
        x = random.randint(0,9)

# measure = pyRAPL.Measurement('bar')
# measure.begin()

fucntion_to_measure()
# measure.end()
# measure.Result()
# measure.export(csv_output)
csv_output.save()