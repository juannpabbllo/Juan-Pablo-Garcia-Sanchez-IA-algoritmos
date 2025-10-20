# Primero instala pgmpy si no lo tienes
# pip install pgmpy

from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference

# Crear DBN simple: variable del tiempo t "Estado" depende del tiempo anterior
dbn = DBN()

# Nodo en t=0
dbn.add_nodes_from([('Estado', 0), ('Observacion', 0)])
# Nodo en t=1
dbn.add_nodes_from([('Estado', 1), ('Observacion', 1)])

# Arcos intra-t (Observación depende del Estado actual)
dbn.add_edge(('Estado', 0), ('Observacion', 0))
dbn.add_edge(('Estado', 1), ('Observacion', 1))

# Arcos inter-t (Estado siguiente depende del Estado anterior)
dbn.add_edge(('Estado', 0), ('Estado', 1))

# Definir tablas de probabilidad condicional (CPT)
from pgmpy.factors.discrete import TabularCPD

cpd_estado_0 = TabularCPD(('Estado', 0), 2, [[0.7], [0.3]])  # 0: Seco, 1: Mojado
cpd_observacion_0 = TabularCPD(('Observacion', 0), 2,
                               [[0.9, 0.2],  # Probabilidades de predicción
                                [0.1, 0.8]],
                               evidence=[('Estado', 0)], evidence_card=[2])
cpd_estado_1 = TabularCPD(('Estado', 1), 2,
                          [[0.8, 0.3],
                           [0.2, 0.7]],
                          evidence=[('Estado', 0)], evidence_card=[2])
cpd_observacion_1 = TabularCPD(('Observacion', 1), 2,
                               [[0.9, 0.2],
                                [0.1, 0.8]],
                               evidence=[('Estado', 1)], evidence_card=[2])

dbn.add_cpds(cpd_estado_0, cpd_observacion_0, cpd_estado_1, cpd_observacion_1)

# Inferencia
dbn_inf = DBNInference(dbn)

# Probabilidad de estado en t=1 dado observación en t=0
query = dbn_inf.forward_inference([('Estado', 1)], evidence={('Observacion', 0): 1})
print("\n Probabilidad de estado en t=1 dado observación en t=0:", query)
