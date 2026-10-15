"""oscattn: attention read out from the phases of coupled oscillators (dynamics from Un-0)."""
from oscattn.tasks import Task, make_task
from oscattn.coupling import continuity_coupling
from oscattn.dynamics import simulate, coupling_matrix
from oscattn.readout import phase_attention, same_object, binding_accuracy, stacked_local_attention

__all__ = ['Task', 'make_task', 'continuity_coupling', 'simulate', 'coupling_matrix', 'phase_attention',
           'same_object', 'binding_accuracy', 'stacked_local_attention']
