from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianNetwork([
    ('D', 'F'),
    ('D', 'C'),
    ('D', 'T'),
    ('D', 'H')
])

cpd_d = TabularCPD(
    variable='D',
    variable_card=2,
    values=[[0.3], [0.7]]
)

cpd_f = TabularCPD(
    variable='F',
    variable_card=2,
    values=[
        [0.1, 0.5],
        [0.9, 0.5]
    ],
    evidence=['D'],
    evidence_card=[2]
)

cpd_c = TabularCPD(
    variable='C',
    variable_card=2,
    values=[
        [0.2, 0.4],
        [0.8, 0.6]
    ],
    evidence=['D'],
    evidence_card=[2]
)

cpd_t = TabularCPD(
    variable='T',
    variable_card=2,
    values=[
        [0.3, 0.7],
        [0.7, 0.3]
    ],
    evidence=['D'],
    evidence_card=[2]
)

cpd_h = TabularCPD(
    variable='H',
    variable_card=2,
    values=[
        [0.4, 0.6],
        [0.6, 0.4]
    ],
    evidence=['D'],
    evidence_card=[2]
)

model.add_cpds(cpd_d, cpd_f, cpd_c, cpd_t, cpd_h)

assert model.check_model(), "Model is invalid"

inference = VariableElimination(model)

q1 = inference.query(variables=['D'], evidence={'F': 1, 'C': 1})
print("Query 1: P(Disease | Fever = Yes, Cough = Yes)")
print(q1)

q2 = inference.query(variables=['D'], evidence={'F': 1, 'C': 1, 'H': 1})
print("\nQuery 2: P(Disease | Fever = Yes, Cough = Yes, Chills = Yes)")
print(q2)

q3 = inference.query(variables=['T'], evidence={'D': 0})
print("\nQuery 3: P(Fatigue = Yes | Disease = Flu)")
print(q3)
