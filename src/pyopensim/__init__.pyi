from typing import Any

# Core modules - always available
from . import actuators as actuators
from . import analyses as analyses
from . import common as common
from . import simbody as simbody
from . import simulation as simulation
from . import tools as tools

# Version is imported from package metadata
__version__: str
__opensim_version__: str

# Optional modules - may not be present
examplecomponents: Any
moco: Any
report: Any

# Re-exported classes from simbody (available at package level)
from .simbody import Vec3 as Vec3
from .simbody import Rotation as Rotation
from .simbody import Transform as Transform
from .simbody import Inertia as Inertia

# Re-exported classes from common (available at package level)
from .common import Component as Component
from .common import Storage as Storage
from .common import Array as Array
from .common import StepFunction as StepFunction
from .common import ConsoleReporter as ConsoleReporter

# Re-exported classes from simulation (available at package level)
from .simulation import Model as Model
from .simulation import Manager as Manager
from .simulation import Body as Body
from .simulation import PinJoint as PinJoint
from .simulation import PhysicalOffsetFrame as PhysicalOffsetFrame
from .simulation import Ellipsoid as Ellipsoid
from .simulation import Millard2012EquilibriumMuscle as Millard2012EquilibriumMuscle
from .simulation import PrescribedController as PrescribedController
from .simulation import InverseKinematicsSolver as InverseKinematicsSolver
from .simulation import InverseDynamicsSolver as InverseDynamicsSolver

# Re-exported classes from actuators (available at package level)
from .actuators import CoordinateActuator as CoordinateActuator
from .actuators import PointActuator as PointActuator

# Re-exported classes from tools (available at package level)
from .tools import InverseKinematicsTool as InverseKinematicsTool
from .tools import InverseDynamicsTool as InverseDynamicsTool
from .tools import ForwardTool as ForwardTool
from .tools import AnalyzeTool as AnalyzeTool
