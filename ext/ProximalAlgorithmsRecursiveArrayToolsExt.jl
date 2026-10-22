module ProximalAlgorithmsRecursiveArrayToolsExt

using ProximalAlgorithms
using RecursiveArrayTools: ArrayPartition

# The reductions over an iterate go part by part, so that each part takes its own method: the loop
# for `Array` storage, a lazy broadcast for a device array. A broadcast over the partition itself
# indexes the elements of every part one by one, which a device array refuses.
ProximalAlgorithms._pogm_restart_sum(w::ArrayPartition, z::ArrayPartition, x::ArrayPartition) =
    sum(map(ProximalAlgorithms._pogm_restart_sum, w.x, z.x, x.x))

end # module
