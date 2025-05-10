from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

model = BayesianNetwork([
    ('I', 'G'),
    ('S', 'G'),
    ('D', 'G'),
    ('G', 'P')
])

cpd_i = TabularCPD(
    variable='I',
    variable_card=2,
    values=[[0.3], [0.7]]
)

cpd_s = TabularCPD(
    variable='S',
    variable_card=2,
    values=[[0.4], [0.6]]
)

cpd_d = TabularCPD(
    variable='D',
    variable_card=2,
    values=[[0.4], [0.6]]
)

cpd_g = TabularCPD(
    variable='G',
    variable_card=3,
    values=[
        [0.1, 0.2, 0.3, 0.5, 0.3, 0.4, 0.6, 0.9],
        [0.3, 0.3, 0.4, 0.3, 0.4, 0.4, 0.3, 0.1],
        [0.6, 0.5, 0.3, 0.2, 0.3, 0.2, 0.1, 0.0]
    ],
    evidence=['I', 'S', 'D'],
    evidence_card=[2, 2, 2]
)

cpd_p = TabularCPD(
    variable='P',
    variable_card=2,
    values=[
        [0.05, 0.2, 0.5],
        [0.95, 0.8, 0.5]
    ],
    evidence=['G'],
    evidence_card=[3]
)

model.add_cpds(cpd_i, cpd_s, cpd_d, cpd_g, cpd_p)

assert model.check_model(), "Model is invalid"

inference = VariableElimination(model)

q1 = inference.query(variables=['P'], evidence={'S': 1, 'D': 0})
print("Query 1: P(Pass = Yes | StudyHours = Sufficient, Difficulty = Hard)")
print(q1)

q2 = inference.query(variables=['I'], evidence={'P': 1})
print("\nQuery 2: P(Intelligence = High | Pass = Yes)")
print(q2)
