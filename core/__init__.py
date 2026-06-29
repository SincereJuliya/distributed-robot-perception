"""
core package — swappable algorithm layer.

Each algorithm is one class in one file. To replace an algorithm, write a new
class with the same public methods and update the import here.
"""
from core.consensus     import GossipConsensus
from core.kalman        import DistributedKalmanFilter
from core.coverage      import LloydVoronoi
from core.graph_metrics import CommGraph
from core.trust         import TrustReputation
from core.formation     import FormationControl
